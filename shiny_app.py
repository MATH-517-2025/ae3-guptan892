#Step 1: Import necessary libraries

from shiny import App, ui, render, reactive
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.stats import beta

#np.random.seed(13)


#Step 2: create regression function
# Regression function: m(x)
def m_func(x):
    return np.sin(1 / (x/3 + 0.1))

#Step 3: Create simulation dataset (covariate X from a beta distribution Beta (α ,β))
def simulate_data(n, alpha, beta_param, sigma_squared):
    X = beta.rvs(alpha, beta_param, size=n) #beta distribution to create covariate X
    epsilon = np.random.normal(0, np.sqrt(sigma_squared), size=n) #noise (normal(0, sigma^2))
    Y = m_func(X) + epsilon
    return X, Y

#Step 4: divide into n-blocks and fit polynomial regression in each block (to estimate m(x) locally in each block)
#Returns a list of fitted models for each block
#Use this to estimate theta22 and sigma^2
def fit_polynomial_n_blocks(X, Y, N_blocks):
    #get sample size and divide into N blocks
    n = len(X) 
    indices = np.array_split(np.arange(n), N_blocks)
    
    #create list to store block models
    block_models = []

    #for each block fit a 4th degree polynomial regression
    for idx in indices:
        #get x and y values for the block
        Xi = X[idx].reshape(-1, 1) #(sklearn needs 2D array for features)
        Yi = Y[idx] 
        
        #convert data into polynomial (degree 4) form - transforms data to [1,x,x^2,x^3,x^4] - so x=0.5 becomes [1,0.5,0.25,0.125,0.0625]
        #need to manually transform X data to polynomial form before fitting linear regression using sklearn
        polynomial_transform = PolynomialFeatures(degree=4) 
        Xi_polynomial_transform = polynomial_transform.fit_transform(Xi)
        
        #create linear regression model and fit to X, Y from the block
        model = LinearRegression()
        model.fit(Xi_polynomial_transform, Yi)
        
        #store this block's model
        block_models.append((model, idx))
    return block_models


#Step 5: Using the fitted models from each block, estimate theta22 and sigma^2

# Estimate theta22 by using second derivative of fitted polynomial in each block
def estimate_theta22(X, block_models):
    n = len(X)
    theta_hat_vals = []

    #for each block model, calculate second derivative estimates and their sum of squares
    for model, idx in block_models:

        #Convert X to polynomial features (degree 4)
        poly = PolynomialFeatures(degree=4)
        Xi = X[idx].reshape(-1, 1)
        Xi_poly = poly.fit_transform(Xi)

        #Get coefficients of the fitted polynomial
        coefs = model.coef_
        b0 = model.intercept_  
        b1 = coefs[1]
        b2 = coefs[2]
        b3 = coefs[3]
        b4 = coefs[4]

        #compute 2nd derivative estimate
        Xi_vals = X[idx]
        #because m(x) = b0 + b1*x + b2*x^2 + b3*x^3 + b4*x^4, m''(x) is below
        m_func_2nd_deriv_est =(2 * b2) + (6 * b3 * Xi_vals) + (12 * b4 * Xi_vals**2)
        
        #theta = sum of squares of 2nd derivative estimates in this block
        theta_hat_vals.append(np.sum(m_func_2nd_deriv_est**2))
    
    #average theta over all blocks
    return np.sum(theta_hat_vals) / n

# Estimate sigma^2 & associated rss
def estimate_sigma_squared(X, Y, block_models):
    n = len(X)
    rss_total = 0

    #for each block model, calculate residual sum of squares and sum them
    for model, idx in block_models:
        Xi = X[idx].reshape(-1, 1) #reshape to 2d for sklearn
        Yi = Y[idx]

        #Convert X to polynomial features (degree 4)
        poly = PolynomialFeatures(degree=4)
        Xi_poly = poly.fit_transform(Xi)
        
        #use the fitted model to predict Y values and calculate residuals
        preds = model.predict(Xi_poly)
        residuals = Yi - preds

        #add this block's RSS to total RSS
        rss_total += np.sum(residuals**2)

    #estimate sigma^2
    sigma_squared_hat = rss_total / (n - 5 * len(block_models))
    return sigma_squared_hat, rss_total


#Step 6: Compute h_AMISE and Mallow's Cp
# Compute h_AMISE
def compute_h_amise(n, sigma_squared_hat, theta22_hat, supp_x=1):
    return (35 * sigma_squared_hat * abs(supp_x) / theta22_hat)**(1/5) * n**(-1/5)

# Compute Mallow's Cp
def compute_mallows_cp(rss_N_blocks, rss_N_block_max, n, N_blocks, N_block_max):
    cp = (rss_N_blocks / (rss_N_block_max / (n - 5 * N_block_max))) - (n - 10 * N_blocks)
    return cp


#Step 7: Run simulations
def run_simulation(n, N_blocks, alpha, beta_param, sigma_squared=1, num_times=200):
    h_amise_list=[]
    cp_list=[]
    for i in range(num_times):
        X, Y = simulate_data(n, alpha, beta_param, sigma_squared) #create data
        block_models = fit_polynomial_n_blocks(X, Y, N_blocks) #Fit polynomial models on N blocks
        
        #estimate theta22 and sigma^2
        theta22_hat = estimate_theta22(X, block_models) 
        sigma_squared_hat, rss_N_blocks = estimate_sigma_squared(X, Y, block_models)
        
        #compute h_AMISE
        h_amise = compute_h_amise(n, sigma_squared_hat, theta22_hat)
        
        # Compute mallow's cp  
        N_block_max = max(min(n // 20, 5), 1) #compute n-max ( Ruppert et al. (1995) )
        block_models_Nmax = fit_polynomial_n_blocks(X, Y, N_block_max) #fit polynomial models on Nmax blocks
        sigma_squared_max, rss_N_block_max = estimate_sigma_squared(X, Y, block_models_Nmax) #get rss of n block max
        cp = compute_mallows_cp(rss_N_blocks, rss_N_block_max, n, N_blocks, N_block_max) #compute CP

        h_amise_list.append(h_amise)
        cp_list.append(cp)
    h_amise= np.sum(h_amise_list)/num_times
    cp= np.sum(cp_list)/num_times
    return h_amise, cp

#step 8: create shiny app
app_ui = ui.page_fluid(
    ui.h2("Polynomial Regression Blocks Simulation"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_slider("sample_size", "Sample size (n):", min=50, max=500, step=50, value=100),
            ui.input_slider("n_blocks", "Number of blocks (N_blocks):", min=1, max=10, step=1, value=2),
            ui.input_slider("alpha", "Beta Distribution Alpha:", min=1, max=10, step=1, value=2),
            ui.input_slider("beta_param", "Beta Distribution Beta:", min=1, max=10, step=1, value=2),
            ui.input_slider("sigma_squared", "Noise variance (sigma^2):", min=0.1, max=5.0, step=0.1, value=1.0),
            ui.input_action_button("run", "Run Simulation"),
        ),

        ui.div(
            ui.output_plot("h_amise_plot"),
            ui.output_text_verbatim("simulation_summary"),
            ui.output_text_verbatim("input_warnings"),
        )
    )
)


def server(input, output, session):
    warnings = reactive.Value(None)

    @reactive.Calc
    def simulation_result():
        input.run()
        n = input.sample_size()
        N_blocks = input.n_blocks()
        alpha = input.alpha()
        beta_param = input.beta_param()
        sigma_squared = input.sigma_squared()

        warning_msgs = []
        if N_blocks > n:
            warning_msgs.append(f"Number of blocks (N_blocks) reduced from {N_blocks} to {n} to avoid empty blocks.")
            N_blocks = n
        if alpha <= 0:
            warning_msgs.append(f"Alpha was <= 0; reset to 1.")
            alpha = 1
        if beta_param <= 0:
            warning_msgs.append(f"Beta parameter was <= 0; reset to 1.")
            beta_param = 1
        if sigma_squared <= 0:
            warning_msgs.append(f"Noise variance (sigma^2) was <= 0; reset to 1.")
            sigma_squared = 1

        warnings.set("\n".join(warning_msgs) if warning_msgs else None)

        h_amise, cp = run_simulation(n, N_blocks, alpha, beta_param, sigma_squared)

        return dict(
            n=n,
            N_blocks=N_blocks,
            alpha=alpha,
            beta_param=beta_param,
            sigma_squared=sigma_squared,
            h_amise=h_amise,
            cp=cp
        )

    @output
    @render.plot
    def h_amise_plot():
        n = input.sample_size()
        N_blocks = input.n_blocks()
        alpha = input.alpha()
        beta_param = input.beta_param()
        sigma_squared = input.sigma_squared()

        block_counts = list(range(1, N_blocks + 1))
        h_amise_values = []

        for blocks in block_counts:
            h_amise, _ = run_simulation(n, blocks, alpha, beta_param, sigma_squared)
            h_amise_values.append(h_amise)

        plt.figure(figsize=(8, 5))
        sns.lineplot(x=block_counts, y=h_amise_values, marker='o')
        plt.xlabel('Number of Blocks')
        plt.ylabel('h_AMISE')
        plt.title(f"h_AMISE vs Number of Blocks (n={n}, α={alpha}, β={beta_param})")
        plt.xticks(block_counts)
        plt.tight_layout()

        return plt.gcf()

    @output
    @render.text
    def simulation_summary():
        result = simulation_result()
        return (
            f"Simulation summary:\n"
            f"Sample size (n): {result['n']}\n"
            f"Number of Blocks (N_blocks): {result['N_blocks']}\n"
            f"Beta distribution alpha: {result['alpha']}\n"
            f"Beta distribution beta: {result['beta_param']}\n"
            f"Noise variance (sigma^2): {result['sigma_squared']}\n\n"
            f"Computed h_AMISE: {result['h_amise']:.4f}\n"
            f"Mallow's Cp: {result['cp']:.4f}"
        )

    @output
    @render.text
    def input_warnings():
        msg = warnings.get()
        return f"⚠️ Input Warnings:\n{msg}" if msg else ""


app = App(app_ui, server)
