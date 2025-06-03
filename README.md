
# instinct-jcas-simulator

Software tool to evaluate the interplay between Communication and Sensing in large scale Joint Communication and Sensing (JCAS) networks.

## About

This simulator is part of the research activities in the **SNS 6G Instinct** European project, focusing on exploring the synergy between communication and sensing in next-generation JCAS networks. For more details, see the full project plan at [Instinct Joint Sensing and Communication](https://www.barkhauseninstitut.org/en/instinct-joint-sensing-and-communication-for-future-connectivity)

## Getting Started

Follow these steps to set up and run the simulator:

### 1. Clone the Repository

Clone this repository to your local machine using:

```
git clone https://github.com/emanuelemengoli/instinct-jcas-simulator.git
cd instinct-jcas-simulator
```

### 2. Create and Activate a Virtual Environment

It is recommended to use a Python virtual environment to manage dependencies.

#### On Unix/macOS:

```
python3 -m venv venv
source venv/bin/activate
```

#### On Windows:

```
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

Install all required Python packages from the requirements file:

```
pip install -r simulation_files/utils/requirements.txt
```

### 4. Run the Simulator

Use the provided Jupyter notebook to run the simulator:

```
jupyter notebook main.ipynb
```

This is a centralized processing environment, allowing users to adjust parameter choices to configure custom simulation scenarios.

## Requirements

* Python 3.7 or higher
* Jupyter Notebook

Make sure Jupyter is installed in your virtual environment. If not, install it using:

```
pip install notebook
```

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for more details.
