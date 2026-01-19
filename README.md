# 📘 Semester Scoreboard Dashboard

A comprehensive academic progress tracking dashboard built with Streamlit for managing semester marks, calculating grades, and visualizing performance across multiple subjects.

## 🚀 Features

- **Multi-Subject Support**: Track marks for Statistics, Machine Learning, Deep Neural Networks, and Mathematics
- **Special EC1 Calculations**: Different calculation methods for each subject
  - **DNN**: EC1 = max(Quiz 1, Quiz 2) + (Assignment 1 + Assignment 2 + Assignment 3) × 5/6
  - **ML**: EC1 = max(Quiz 1, Quiz 2) + Assignment 1 + Assignment 2  
  - **Statistics & Math**: EC1 = Quiz 1 + Quiz 2 + Assignment 1 + Assignment 2
- **Real-time Grade Calculation**: Automatic computation of grades and CGPA
- **Interactive Data Editors**: Easy-to-use forms for entering marks
- **Performance Visualization**: Charts showing academic progress
- **Persistent Data Storage**: CSV-based data storage for tracking over time

## 📊 Grade System

- **A Grade**: 75+ marks (10 points)
- **B Grade**: 65+ marks (8 points)  
- **C Grade**: 55+ marks (6 points)
- **D Grade**: 45+ marks (5 points)
- **F Grade**: Below 45 marks (0 points)

## 🛠️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd semester-scoreboard
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

## 📁 File Structure

```
├── app.py              # Main Streamlit application
├── marks.csv           # Data storage (auto-created)
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## 💻 Usage

1. **Enter Marks**: Use the subject-specific editors to input quiz, assignment, mid-semester, and final exam scores
2. **View Summary**: Check the semester summary table for calculated totals and grades
3. **Monitor Progress**: View performance charts and track your CGPA
4. **Save Changes**: Click the save button to persist your data

## 🌐 Deployment

This application is optimized for deployment on:
- [Streamlit Cloud](https://streamlit.io/cloud)
- Heroku
- Any platform supporting Python/Streamlit applications

### Deploy to Streamlit Cloud
1. Fork this repository
2. Connect your GitHub account to Streamlit Cloud
3. Deploy directly from your repository

## 📋 Requirements

- Python 3.7+
- Streamlit
- Pandas
- Altair

## 🤝 Contributing

Feel free to fork this project and submit pull requests for improvements!

## 📄 License

This project is open source and available under the MIT License.

---
**Built with ❤️ using Streamlit**
- Bar chart of grand totals

## Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Data Storage
- The app saves to a local file named `marks.csv` in the same folder as the app.
- No cloud or external network calls are made.

## Notes
- To import marks, use the CSV upload control and provide a file with the same columns as the app's exported CSV.
- If you rename subjects, ensure column alignment before importing.
