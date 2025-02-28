# 🚀 Crypto Predictor Pro

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

<p align="center">
  <img src="capture1.png" alt="Crypto Predictor Pro Banner" width="600">
</p>

## 🌟 Features

- 📊 Real-time cryptocurrency price predictions using LSTM neural networks
- 📈 Advanced technical analysis with multiple indicators
- 🔄 Smart caching system for optimal performance
- 🎯 Custom trading strategy scanner
- 📱 Responsive web interface powered by Streamlit
- 🌡️ Market sentiment analysis with Fear & Greed Index

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/HAMMOUDAmustaphaahmed/Crypto_Predictor.git

# Create and activate virtual environment
python -m venv crypto_env
source crypto_env/bin/activate  # Linux/Mac
# or
crypto_env\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run crypto_predictor.py --logger.level=error
```

## 📊 Technical Features

### Deep Learning Model
- Bidirectional LSTM with batch normalization
- Adaptive learning rate with early stopping
- Mixed precision training for better performance

### Technical Indicators
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume Analysis

### Data Processing
- Real-time data from Binance API
- Intelligent caching system
- Advanced error handling
- Automatic data normalization

## 🎯 Trading Strategy Scanner

Our custom scanner looks for specific patterns in the market:
- Green candle analysis
- Volume profile
- Price action patterns
- Market structure analysis

## 📱 User Interface

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="capture2.png" alt="Price Predictor" width="400"/><br/>
        <em>Price Prediction Interface</em>
      </td>
      <td align="center">
        <img src="path_to_screenshot2.png" alt="Strategy Scanner" width="400"/><br/>
        <em>Strategy Scanner Interface</em>
      </td>
    </tr>
  </table>
</div>

## 📋 Requirements

```plaintext
numpy>=1.19.2
pandas>=1.3.0
streamlit>=1.0.0
scikit-learn>=0.24.2
tensorflow>=2.6.0
plotly>=5.1.0
requests>=2.26.0
joblib>=1.0.1
```

## 🔧 Configuration

The application supports various configuration options:
- Multiple timeframes (15m to 1M)
- Customizable prediction periods
- Adjustable model parameters
- Cache management settings

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 🙏 Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [Streamlit](https://streamlit.io/)
- [Binance API](https://binance-docs.github.io/apidocs/)
- [Alternative.me API](https://alternative.me/crypto/fear-and-greed-index/)

## ⚠️ Disclaimer

This software is for educational purposes only. Cryptocurrency trading carries a high level of risk, and may not be suitable for everyone. Do not trade with money you cannot afford to lose.

---

<div align="center">
  Made with ❤️ by HAMMOUDAmustaphaahmed
  <br>
  <a href="https://github.com/HAMMOUDAmustaphaahmed">GitHub</a> •
  <a href="https://www.linkedin.com/in/hammouda-ahmed-mustapha-a55270195/">LinkedIn</a>
</div>
