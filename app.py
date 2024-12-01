import pandas as pd
import numpy as np
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

# Streamlit visualization
st.sidebar.title("Portfolio Optimization Overview")
choice = st.sidebar.radio("Steps", ["Client","Data Exploration", "ERC Portfolio Optimization", "Efficient Frontier", "Chat"])
st.sidebar.image("https://www.leparisien.fr/resizer/Uygcq2JNnlfC_sJQnmRoqb3l7yo=/932x582/cloudfront-eu-central-1.images.arcpublishing.com/leparisien/UF2NA7NXPVANHPTUCEZE6CMNUE.jpg")
st.sidebar.info("Select a step from the dropdown menu to explore the data, optimize the portfolio, or analyze the efficient frontier.")

# Preprocess data
def preprocess_data(df, start_date='1992-02-29'):
    df.index = pd.to_datetime(df.index, errors='coerce').to_period('M').to_timestamp('M')
    return df[df.index >= start_date].dropna()

def get_close_prices(df):
    return df['close'] if 'close' in df.columns else df['Close']

# Load data
CMDTY = pd.read_csv('commodity-prices.csv', index_col=0)
CMDTY = CMDTY.iloc[:, 9:]

SMI = pd.read_csv('SMI.csv', index_col=0)
SMI = get_close_prices(SMI)

SPX = pd.read_csv('SPX.csv', index_col=0)
SPX = get_close_prices(SPX)

USDCHF = pd.read_csv('USDCHF.csv', index_col=0)
USDCHF = get_close_prices(USDCHF)

CMDTY = preprocess_data(CMDTY)
SMI = preprocess_data(SMI)
SPX = preprocess_data(SPX)
USDCHF = preprocess_data(USDCHF)

# Calculate simple returns and annualize (assuming monthly data)
CMDTY_returns = CMDTY.pct_change().dropna()
CMDTY_log_returns = np.log(CMDTY / CMDTY.shift(1)).dropna()

SMI_returns = SMI.pct_change().dropna()
SMI_log_returns = np.log(SMI / SMI.shift(1)).dropna()

SPX_returns = SPX.pct_change().dropna()
SPX_log_returns = np.log(SPX / SPX.shift(1)).dropna()

USDCHF_returns = USDCHF.pct_change().dropna()
USDCHF_log_returns = np.log(USDCHF / USDCHF.shift(1)).dropna()

# Combine data
common_dates = CMDTY.index.intersection(SMI.index).intersection(SPX.index).intersection(USDCHF.index)
CMDTY_returns = CMDTY_returns.reindex(common_dates).dropna()
SMI_returns = SMI_returns.reindex(common_dates).dropna()
SPX_returns = SPX_returns.reindex(common_dates).dropna()
USDCHF_returns = USDCHF_returns.reindex(common_dates).dropna()

# Combine portfolio returns
commodity_portfolio_returns = np.mean(CMDTY_returns, axis=1)
equity_returns = pd.concat([pd.Series(SPX_returns, index=common_dates), pd.Series(SMI_returns, index=common_dates)], axis=1).dropna()
fx_portfolio_returns = pd.Series(USDCHF_returns, index=common_dates).dropna()

if choice == "Efficient Frontier":
    st.title("Efficient Frontier")
    st.markdown("This page shows the efficient frontier based on data and a user-entered risk-free rate. The Markowitz Optimization method is used to optimize the equity portfolio. The efficient frontier is calculated based on the expected returns and volatilities of the assets. The user can adjust the risk-free rate and explore the efficient frontier with the tangency portfolio. The tangency portfolio is the optimal portfolio that maximizes the Sharpe ratio. The page also includes a portfolio optimization section where the user can adjust the risk-free rate, gamma, and short selling constraints to optimize the portfolio.")
    # Markowitz Optimization for Equity Portfolio
    SPX_annualized_mu = np.array([(SPX_returns.mean() + 1) ** 12 - 1])
    SMI_annualized_mu = np.array([(SMI_returns.mean() + 1) ** 12 - 1])

    # Concatenate the 1D arrays for expected returns
    equity_mu = np.concatenate((SPX_annualized_mu, SMI_annualized_mu))

    SPX_annualized_vol = np.array([SPX_returns.std()]) * np.sqrt(12)
    SMI_annualized_vol = np.array([SMI_returns.std()]) * np.sqrt(12)  # Volatility for SMI

    equity_vol = np.concatenate((SPX_annualized_vol, SMI_annualized_vol))

    equity_returns = pd.concat([SPX_returns, SMI_returns], axis=1)
    equity_correl_matrix = equity_returns.corr().values

    # Covariance matrix using np.outer for efficiency
    equity_cov_matrix = np.outer(equity_vol, equity_vol) * equity_correl_matrix  # Covariance matrix

    equity_x0 = np.array([1 / equity_returns.shape[1]] * equity_returns.shape[1])


    def QP(x, sigma, mu, gamma):
        v = 0.5 * x.T @ sigma @ x - gamma * x.T @ mu  # Quadratic function to minimize
        return v


    def efficient_frontier(gam, constraints, equity_mu, equity_cov_matrix):
        res = minimize(QP, equity_x0, args=(equity_cov_matrix, equity_mu, gam), options={'disp': False},
                       constraints=constraints)
        equity_optimized_weights = res.x

        # Ensure equity_mu is a 1D array
        equity_mu = equity_mu.flatten()  # Flatten in case it is 2D

        # Expected return of the optimized portfolio
        mu_optimized = equity_optimized_weights @ equity_mu

        # Portfolio volatility (using matrix multiplication)
        vol_optimized = np.sqrt(equity_optimized_weights @ equity_cov_matrix @ equity_optimized_weights)
        return mu_optimized, vol_optimized, equity_optimized_weights


    # Constraints for optimization (weights sum to 1)
    constraints = [LinearConstraint(np.ones(equity_x0.shape), ub=1), LinearConstraint(-np.ones(equity_x0.shape), ub=-1)]

    # Efficient frontier calculation
    mu_efficient_frontier = []
    vol_efficient_frontier = []
    weights_efficient_frontier = []

    for gam in np.linspace(-0.5, 3, 501):
        mu, vol, _ = efficient_frontier(gam, constraints, equity_mu, equity_cov_matrix)
        mu_efficient_frontier.append(mu)
        vol_efficient_frontier.append(vol)

    ticker = ['SPX', 'SMI']
    for i in range(len(ticker)):
        plt.annotate(ticker[i], (equity_vol[i], equity_mu[i]))

    chart_data = pd.DataFrame(
        {'vol': vol_efficient_frontier, 'mu': mu_efficient_frontier, 'color': '#0000FF', 'size': 2})

    # Risk-free rate slider
    rf = st.slider('Risk free rate?', 0.0, 0.05, step=0.005)
    st.session_state['rf'] = rf

    with st.expander("Tangency Portfolio"):
        # Tangency Portfolio Weights
        weights_tangency = (np.linalg.inv(equity_cov_matrix) @ (equity_mu - rf)) / (
                np.ones(equity_cov_matrix.shape[0]) @ np.linalg.inv(equity_cov_matrix) @ (equity_mu - rf))
        st.session_state['weights_tangency'] = weights_tangency
        mu_tangency = weights_tangency @ equity_mu
        vol_tangency = np.sqrt(weights_tangency @ equity_cov_matrix @ weights_tangency)

        # Add Tangency Portfolio to chart
        adding_rf = {'vol': 0, 'mu': rf, 'color': '#FFA500', 'size': 10}
        chart_data = pd.concat([chart_data, pd.DataFrame([adding_rf])], ignore_index=True)
        adding_tangency = {'vol': vol_tangency, 'mu': mu_tangency, 'color': '#FF0000', 'size': 10}
        chart_data = pd.concat([chart_data, pd.DataFrame([adding_tangency]), pd.DataFrame([adding_rf])],
                               ignore_index=True)

        # Plot the efficient frontier
        st.scatter_chart(chart_data, x='vol', y='mu', color='color', size='size')

        # Efficient Frontier Plot with Tangency Portfolio
        chart_data = pd.DataFrame({
            'Volatility': vol_efficient_frontier,
            'Return': mu_efficient_frontier,
            'Type': ['Frontier'] * len(vol_efficient_frontier)
        })
        # Add individual assets and tangency portfolio to the chart
        asset_points = pd.DataFrame({"Volatility": equity_vol, "Return": equity_mu, "Type": ticker})
        tangency_point = pd.DataFrame({"Volatility": [vol_tangency], "Return": [mu_tangency], "Type": ["Tangency"]})

        # Combine all data
        chart_data = pd.concat([chart_data, asset_points, tangency_point], ignore_index=True)

        # Plot the efficient frontier
        st.write("### Efficient Frontier with Risk-Free Asset")
        st.scatter_chart(chart_data, x='Volatility', y='Return', color='Type')

    with st.expander("Summary Statistics"):
        st.write("### Summary Statistics of Tangency Portfolio")
        summary_stats = pd.DataFrame({
            "Metric": ["Expected Return", "Volatility", "Sharpe Ratio", "Skewness", "Kurtosis", "Max Drawdown", "Weights"],
            "Value": [mu_tangency, vol_tangency, (mu_tangency - rf) / vol_tangency, stats.skew(np.dot(equity_returns, weights_tangency)), stats.kurtosis(np.dot(equity_returns, weights_tangency)), np.dot(equity_returns, weights_tangency).min(), weights_tangency]
        })
        st.table(summary_stats)


    # Portfolio Optimization with Short Selling Toggle
    with st.expander("Portfolio Optimization"):
        st.info("This page shows the efficient frontier based on data, the user-entered risk-free rate, and gamma")

        if 'rf' in st.session_state:
            rf = st.session_state.rf
            weights_tangency = st.session_state.weights_tangency
            mu_mod = np.append(equity_mu, rf)
            vol_mod = np.append(equity_vol, 0)
            cov_matrix_mod = np.zeros([equity_cov_matrix.shape[0] + 1, equity_cov_matrix.shape[0] + 1])
            cov_matrix_mod[:-1, :-1] = equity_cov_matrix
            x0_mod = np.array([0.33, 0.33, 0.33])  # Initial guess for weights

            on = st.toggle('Short selling allowed')
            gam = st.slider('Gamma?', 0.0, 1.0, step=0.1)

            if on:
                constraints = [LinearConstraint(np.ones(x0_mod.shape), ub=1),
                               LinearConstraint(-np.ones(x0_mod.shape), ub=-1)]
                res = minimize(QP, x0_mod, args=(cov_matrix_mod, mu_mod, gam), options={'disp': False},
                               constraints=constraints)
                equity_optimized_weights = res.x
                st.bar_chart(pd.DataFrame(equity_optimized_weights, index=ticker + ['risk-free']))

            else:
                constraints = [LinearConstraint(np.ones(x0_mod.shape), ub=1),
                               LinearConstraint(-np.ones(x0_mod.shape), ub=-1),
                               LinearConstraint(np.eye(x0_mod.shape[0]), lb=0)]
                res = minimize(QP, x0_mod, args=(cov_matrix_mod, mu_mod, gam), options={'disp': False},
                               constraints=constraints)
                equity_optimized_weights = res.x

                fig1, ax1 = plt.subplots()
                ax1.pie(np.abs(equity_optimized_weights), labels=ticker + ['risk-free'], autopct='%1.1f%%',
                        startangle=90)
                ax1.axis('equal')
                st.pyplot(fig1)
        else:
            st.write('You must select the risk-free rate in the tab "Efficient frontier" first.')
        with st.chat_message("user"):
            st.write("Are you willing to invest?📈")

# Update and annualize returns
updated_returns = pd.Series(
    [
        (1 + commodity_portfolio_returns.mean()),  # already annualized Commodity returns
        (1 + equity_returns.mean(axis=1).mean()),  # already annualized Equity returns
        (1 + fx_portfolio_returns.mean())         # already annualized FX returns
    ],
    index=["Commodity", "Equity", "FX"]
)

# Calculate the annualized covariance matrix
updated_cov_matrix = pd.DataFrame(
    {
        "Commodity": [commodity_portfolio_returns.var(), commodity_portfolio_returns.cov(equity_returns.mean(axis=1)), commodity_portfolio_returns.cov(fx_portfolio_returns)],
        "Equity": [equity_returns.mean(axis=1).cov(commodity_portfolio_returns), equity_returns.mean(axis=1).var(), equity_returns.mean(axis=1).cov(fx_portfolio_returns)],
        "FX": [fx_portfolio_returns.cov(commodity_portfolio_returns), fx_portfolio_returns.cov(equity_returns.mean(axis=1)), fx_portfolio_returns.var()]
    },
    index=["Commodity", "Equity", "FX"]
)   # already annualized covariance matrix

# Calculate the correlation matrix
updated_correl_matrix = updated_cov_matrix.corr()

# Display matrices for debugging or use elsewhere
print("Annualized Returns:")
print(updated_returns)

print("\nUpdated Covariance Matrix (Annualized):")
print(updated_cov_matrix)

print("\nUpdated Correlation Matrix:")
print(updated_correl_matrix)

# ERC Portfolio Optimization
n_assets = updated_returns.shape[0]
initial_weights = np.array([1 / n_assets] * n_assets)

def QP(y, sigma): return 0.5 * y.T @ sigma @ y
logy = lambda y: np.sum(np.log(y))
constraints = [LinearConstraint(np.eye(n_assets), lb=0, ub=1), NonlinearConstraint(logy, lb=-n_assets * np.log(n_assets) - 2, ub=np.inf)]

result = minimize(QP, initial_weights, args=(updated_cov_matrix,), method='trust-constr', constraints=constraints)
optimized_weights_advanced = result.x / np.sum(result.x)
abs_risk_contrib_advanced = optimized_weights_advanced * (updated_cov_matrix @ optimized_weights_advanced) / np.sqrt(optimized_weights_advanced @ updated_cov_matrix @ optimized_weights_advanced)
rel_risk_contrib_advanced = abs_risk_contrib_advanced / np.sqrt(optimized_weights_advanced @ updated_cov_matrix @ optimized_weights_advanced)

if choice == "Client":
    st.title("Client Information")
    st.markdown("""
    This page provides an overview of the client's portfolio and investment preferences. 
    The client is interested in optimizing their portfolio using the Equal Risk Contribution (ERC) method.
    """)

    # Expander for portfolio overview
    with st.expander("📋 Portfolio Overview"):
        st.subheader("Portfolio Components")
        st.write("""
        - **Commodity Investments**: Includes gold, oil, and other commodities.
        - **Equity Indices**: Global indices such as S&P 500 and SMI.
        - **Currency Exchange Rates**: Forex pairs like USD/CHF.
        """)

        st.subheader("Portfolio Preferences")
        st.write("""
        - **Risk Management**: Focus on balancing risk across portfolio components.
        - **Diversification**: Interest in spreading investments across different asset classes.
        - **Optimization Goal**: Use the Equal Risk Contribution (ERC) method to create an efficient portfolio.
        - **Flexibility**: Open to exploring alternative optimization methods.
        """)

    # Portfolio analysis section
    with st.expander("📊 Portfolio Analysis"):
        st.subheader("Analysis Goals")
        st.write("""
        - **Data Exploration**: Analyze the historical performance of portfolio components.
        - **Optimization**: Achieve a balanced risk contribution using ERC.
        - **Efficient Frontier**: Understand trade-offs between risk and return.
        - **Tangency Portfolio**: Explore the highest Sharpe ratio portfolio.
        """)
        st.write("Additional features include an **interactive chatbot** for questions and feedback.")

    # Next steps section
    with st.expander("🚀 Next Steps"):
        st.subheader("Steps to Achieve Client Goals")
        st.write("""
        - **Review Data**: Perform in-depth analysis of portfolio components.
        - **Implement Optimization**: Use ERC to optimize the portfolio.
        - **Visualize**: Explore the efficient frontier and tangency portfolio with interactive charts.
        - **Feedback**: Interact with the chatbot for tailored advice.
        """)

    # Interactive elements
    with st.expander("🔢 Example"):
        st.subheader("Interactive Client Inputs")
        initial_investment = st.number_input("Enter the initial investment amount ($):", min_value=1000, step=500)
        risk_preference = st.slider("Select your risk preference (0 = low, 1 = high):", 0.0, 1.0, 0.5)

        st.write(f"### Portfolio Overview for Initial Investment of ${initial_investment:,.2f}")
        st.write(f"Risk preference: {risk_preference:.2f}")

        # Predefined proportions (you can adjust these as desired)
        default_proportions = {
            "Commodities": 0.3,
            "Equities": 0.5,
            "Forex": 0.2,
        }

        # Input from the user, with defaults
        proportion_commodities = st.number_input(
            "Enter the proportion of Commodities:",
            min_value=0.0, max_value=1.0, step=0.01, value=default_proportions["Commodities"]
        )
        proportion_equities = st.number_input(
            "Enter the proportion of Equities:",
            min_value=0.0, max_value=1.0, step=0.01, value=default_proportions["Equities"]
        )
        proportion_forex = st.number_input(
            "Enter the proportion of Forex:",
            min_value=0.0, max_value=1.0, step=0.01, value=default_proportions["Forex"]
        )

        # Ensure proportions sum to 1 (optional validation)
        if proportion_commodities + proportion_equities + proportion_forex != 1.0:
            st.warning("Proportions must sum to 1. Adjust the values.")

        # Create a DataFrame
        portfolio_data = pd.DataFrame({
            "Asset Class": ["Commodities", "Equities", "Forex"],
            "Proportion": [proportion_commodities, proportion_equities, proportion_forex],
            "Value": [
                initial_investment * proportion_commodities,
                initial_investment * proportion_equities,
                initial_investment * proportion_forex,
            ],
        })

        # Display the DataFrame
        st.write(portfolio_data)

        tab1, tab2 = st.tabs(["Visualization", "Data Table"])
        # Pie chart visualization
        tab1.subheader("Current Portfolio Allocation")
        fig, ax = plt.subplots()
        ax.pie(portfolio_data["Proportion"], labels=portfolio_data["Asset Class"], autopct='%1.1f%%', startangle=90)
        ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
        tab1.pyplot(fig)

        # Data table
        tab2.write("### Portfolio Allocation Details")
        tab2.dataframe(portfolio_data)

        # Interactive chatbot placeholder
        st.subheader("💬 Chatbot Assistance")
        st.markdown("Use the chatbot below for questions or feedback located on the sidebar ;) !")

if choice == "Data Exploration":
    st.title("Data Exploration")
    st.markdown("This page shows the data exploration and analysis of the portfolio components. The data includes commodity prices, equity indices, and currency exchange rates. The analysis includes monthly returns, cumulative returns, and covariance matrices for the portfolio components. The data exploration section provides insights into the historical performance of the portfolio components and their relationships. One thing to be aware of is that the data is monthly and may not reflect the most recent market conditions. Moreover, it covers the period starting from 1992 until mid-2017, which many Crisis have been held.")
    st.subheader("Data Exploration")
    with st.expander("Detailed Commodity Analysis"):
        st.subheader("Commodity monthly returns")
        tab1, tab2 = st.tabs(["Monthly Commodity returns", "Cumul. Sum of Monthly Commodity Returns"])
        tab1.area_chart(CMDTY_returns)
        tab1.write(CMDTY_returns)
        tab2.write('Cumulative sum of monthly return')
        tab2.line_chart(CMDTY_log_returns.cumsum())

    with st.expander("Detailed SPX Analysis"):
        st.subheader("SPX monthly returns")
        tab1, tab2 = st.tabs(["Monthly SPX returns", "Cumul. Sum of Monthly SPX Returns"])
        tab1.area_chart(SPX_returns)
        tab1.write(SPX_returns)
        tab2.write('Cumulative sum of monthly return')
        tab2.line_chart(SPX_log_returns.cumsum())

    with st.expander("Detailed SMI Analysis"):
        st.subheader("SMI monthly returns")
        tab1, tab2 = st.tabs(["Monthly SMI returns", "Cumul. Sum of Monthly SMI Returns"])
        tab1.area_chart(SMI_returns)
        tab1.write(SMI_returns)
        tab2.write('Cumulative sum of monthly return')
        tab2.line_chart(SMI_log_returns.cumsum())

    with st.expander("Detailed USDCHF Analysis"):
        st.subheader("USDCHF monthly returns")
        tab1, tab2 = st.tabs(["Monthly FX returns", "Cumul. Sum of Monthly FX Returns"])
        tab1.area_chart(fx_portfolio_returns)
        tab1.write(fx_portfolio_returns)
        tab2.write('Cumulative sum of monthly return')
        tab2.line_chart(USDCHF_log_returns.cumsum())

    with st.expander("Detailed Covariance Analysis"):
        tab1, tab2 = st.tabs(["Covariance Matrix", "Correlation Matrix"])
        tab1.subheader("Covariance Matrix")
        tab1.table(pd.DataFrame(updated_cov_matrix, index=["Commodity", "Equity", "FX"], columns=["Commodity", "Equity", "FX"]))

        tab2.subheader("Correlation Matrix")
        correlation_matrix = pd.DataFrame(updated_cov_matrix, index=["Commodity", "Equity", "FX"], columns=["Commodity", "Equity", "FX"]).corr()
        fig, ax = plt.subplots()
        sns.heatmap(updated_correl_matrix, annot=True, cmap="coolwarm", ax=ax)
        tab2.pyplot(fig)

    st.subheader("Additional Features")
    if st.checkbox("Show Data Summary"):
        st.write("### Data Summary")
        tab1, tab2, tab3, tab4 = st.tabs(["Commodity", "  SMI  ", "  SPX  ", "USDCHF"])

        tab1.write("#### Commodity Data Summary")
        tab1.write(CMDTY_returns.describe())

        tab2.write("#### SMI Data Summary")
        tab2.write(SMI_returns.describe())

        tab3.write("#### SPX Data Summary")
        tab3.write(SPX_returns.describe())

        tab4.write("#### USDCHF Data Summary")
        tab4.write(USDCHF_returns.describe())

    if st.checkbox("Show Pairplot"):
        st.write("### Pairplot of Returns")
        combined_returns = pd.concat([pd.Series(commodity_portfolio_returns, index=CMDTY.index), equity_returns.mean(axis=1), pd.Series(fx_portfolio_returns, index=USDCHF.index)], axis=1)
        combined_returns.columns = ["Commodity", "Equity", "FX"]
        sns.pairplot(combined_returns)
        st.pyplot()

    sentiment_mapping = [":material/thumb_down:", ":material/thumb_up:"]
    selected = st.feedback("thumbs")
    if selected is not None:
        st.markdown(f"You selected: {sentiment_mapping[selected]}")
        if selected == 1:
            st.balloons()

if choice == "ERC Portfolio Optimization":
    st.title("Equal Risk Contribution Portfolio Optimization")
    st.markdown(""" **Equal Risk Contribution (ERC) Optimization:** The ERC portfolio aims to equalize the risk contributions from its different components. This method was introduced by Maillard, Roncalli, and Teiletche in 2008. The ERC portfolio is considered a middle-ground alternative to an equally weighted portfolio
    and a minimum variance portfolio, balancing risk and weights effectively. The optimization process ensures that each asset contributes equally to the overall portfolio risk. **Citation:** Maillard, S., Roncalli, T., & Teiletche, J. (2008).
    Equally-weighted Risk Contributions: A New Method to Build Risk Balanced Diversified Portfolios. """)
    st.subheader("Equal Risk Contribution Portfolio")
    st.write("Optimized ERC Portfolio Weights")
    st.bar_chart(pd.DataFrame(optimized_weights_advanced, index=["Commodity", "Equity", "FX"], columns=["Weight"]))
    tab1, tab2 = st.tabs(["Absolute RC", "Relative RC"])

    # Absolute Risk Contributions
    tab1.write("Absolute Risk Contributions:")
    abs_risk_contrib_df = pd.DataFrame(
        abs_risk_contrib_advanced,
        index=["Commodity", "Equity", "FX"],
        columns=["Absolute Risk Contribution"]
    )
    tab1.table(abs_risk_contrib_df)
    tab1.bar_chart(abs_risk_contrib_df)

    # Relative Risk Contributions
    tab2.write("Relative Risk Contributions:")
    rel_risk_contrib_df = pd.DataFrame(
        rel_risk_contrib_advanced,
        index=["Commodity", "Equity", "FX"],
        columns=["Relative Risk Contribution"]
    )
    tab2.table(rel_risk_contrib_df)
    tab2.bar_chart(rel_risk_contrib_df)
    sentiment_mapping = ["one", "two", "three", "four", "five"]
    selected = st.feedback("stars")
    if selected is not None:
        st.markdown(f"You selected {sentiment_mapping[selected]} star(s).")

        if selected == 4 or selected == 5:
            st.balloons()

if choice == "Chat":
    st.title("Interactive Chat")
    st.balloons()
    st.markdown("This page allows you to interact with a chatbot. You can ask questions, provide feedback, or request assistance. The chatbot will respond to your messages and provide information or guidance. Feel free to start a conversation and explore the chatbot's capabilities.")
    if "messages" not in st.session_state:
        st.session_state.messages = []


    # Function to display chat messages
    def display_chat():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])


    # Function to handle user input and generate a bot response
    def process_message(user_message):
        # Append user's message to the chat history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Simulate bot response (this is where you could integrate AI or other logic)
        bot_response = f"Bot: You said: '{user_message}'. How can I assist you further?"
        st.session_state.messages.append({"role": "assistant", "content": bot_response})


    # Display chat history
    display_chat()

    # User input form for new message
    user_message = st.text_input("Enter your message:", key="chat_input")

    # Process the message when the user submits it
    if user_message and "submitted" not in st.session_state:
        st.session_state.submitted = True
        process_message(user_message)
        st.rerun()

    with st.expander("Chatbot Instructions"):
        st.write("### Chatbot Instructions")
        st.write("""
        - **Enter Message**: Type your message in the text box.
        - **Submit**: Press 'Enter' or click the 'Submit' button to send your message.
        - **Conversation**: View the chat history to see messages from the chatbot.
        - **Feedback**: Provide feedback or ask questions to interact with the chatbot.
        """)
    if st.button("Submit"):
        st.session_state.submitted = True
        process_message(user_message)
        st.rerun()
    with st.expander("meme"):
        st.title("jk")
        tab1, tab2 = st.tabs(["Meme", "Video"])
        tab1.image("https://ih1.redbubble.net/image.5602107707.2187/st,small,507x507-pad,600x600,f8f8f8.jpg")
        tab2.video("https://www.youtube.com/watch?v=-l0HFgfDWec&pp=ygUpaSdtIGxvb2tpbmcgZm9yIGEgbWFuIGluIGZpbmFuY2Ugb3JpZ2luYWw%3D")
