```markdown
# Numerical RF De-embedding

## 1. Project Overview

This project demonstrates a fundamental process in Radio Frequency (RF) engineering known as **De-embedding**. When measuring high-frequency components using a Vector Network Analyzer (VNA), the Device Under Test (DUT) can rarely be connected directly to the instrument. Instead, it requires cables, connectors, and PCB pads (fixtures).

Unfortunately, these fixtures introduce their own parasitic effects—such as signal attenuation, phase rotation, and parasitic capacitance—which distort the measurement. The goal of this project is to:
1. Synthesize realistic RF measurement data that simulates both an ideal DUT and the parasitic distortions of measurement fixtures.
2. Apply a rigorous mathematical framework (using **ABCD matrices**) to computationally "remove" the fixture's influence.
3. Recover the true, intrinsic characteristics of the DUT.

This project combines Python programming, matrix algebra, and RF engineering concepts (S-parameters, Smith Charts) to solve a real-world signal integrity problem.

---

## 2. Project Structure and File Breakdown

The repository is divided into source code (`src/`) for the mathematical engines, and Jupyter Notebooks (`notebooks/`) for data analysis and visualization.

### 2.1 The Core Logic (`src/` Directory)

#### `src/rf_measurement_generator.py`
This script acts as our "Virtual VNA". It mathematically synthesizes realistic S-parameter datasets. It generates the intrinsic ideal device behavior, models the parasitic fixtures, cascades them, and injects a Gaussian noise floor. 

The generator can model two main topologies. Here is the visual breakdown of what is being simulated (distinguishing the DUT from the parasitic structures):

**A) Shunt Topology (Notch Filter) - *Primary Focus of the Project***
```text
       [ Fixture A ]       [ Intrinsic DUT ]       [ Fixture B ]
Port 1 ----+-----------------------+-----------------------+---- Port 2
           |                       |                       |
         [PAD A]                 [RLC]                   [PAD B]
      (1kΩ || 1pF)            (1Ω-1nH-1pF)            (1kΩ || 1pF)
           |                       |                       |
Ground ----+-----------------------+-----------------------+---- Ground

```

* **Intrinsic DUT:** The series RLC branch in the middle. At resonance (~5.03 GHz), its impedance drops to $1\,\Omega$, shunting the signal to ground and creating a deep "Notch" in transmission.
* **Fixtures (Parasitics):** The PADs on the left and right. They leak high-frequency signals to ground, causing broadband signal loss and shifting the resonant frequency.

**B) Series Topology (Band-pass Filter)**

```text
       [ Fixture A ]                           [ Fixture B ]
Port 1 ----+--------------[ R ]-[ L ]-[ C ]----------+---- Port 2
           |             [ Intrinsic DUT ]           |
         [PAD A]                                   [PAD B]
           |                                         |
Ground ----+-----------------------------------------+---- Ground

```

* **Intrinsic DUT:** Placed directly in the signal path. It blocks out-of-band signals and allows the resonant frequency to pass through.

#### `src/deembedder.py`

This is the mathematical engine of the project. It uses the `scikit-rf` library to perform the actual de-embedding. Because RF components in a series cascade cannot be simply subtracted, the script converts the measured S-parameters into **ABCD (Transmission) matrices**.

The script performs the following surgical matrix operation to isolate the DUT:


$$[T_{DUT}] = [T_{FixA}]^{-1} \cdot [T_{Total}] \cdot [T_{FixB}]^{-1}$$


Once isolated, it converts the result back into S-parameters.

---

### 2.2 The Analysis (`notebooks/` Directory)

The notebooks guide the user step-by-step through the physics, the data, and the final solution:

* **`01_intro_rf_deembedding.ipynb`**: Serves as the setup stage. It explains the theoretical background of the experiment, defines the physical parameters of the synthetic models, and runs `rf_measurement_generator.py` to populate our raw datasets.
* **`02_data_explorator.ipynb`**: Focuses on Exploratory Data Analysis (EDA) of the RF data. It visualizes the raw S-parameters (Insertion Loss $S_{21}$ and Return Loss $S_{11}$). This notebook highlights the physical problem: showing how the fixtures attenuate the signal and severely distort the expected notch filter response.
* **`03_smith_chart.ipynb`**: The culmination of the project. It introduces the **Smith Chart** to visualize complex impedance and phase delays (the "chaotic spiral" caused by cables). It then executes the `deembedder.py` algorithm live, proving mathematically that the de-embedded data successfully un-winds the spiral and perfectly matches the "Golden" (ideal) DUT response.

---

## 3. Installation and Usage

To run this project locally, clone the repository and install the dependencies:

```bash
# Clone the repository
git clone [https://github.com/your-username/numerical-rf-deembedding.git](https://github.com/your-username/numerical-rf-deembedding.git)
cd numerical-rf-deembedding

# Install required Python packages
pip install -r requirements.txt

```

*Main dependencies include: `numpy`, `pandas`, `matplotlib`, and `scikit-rf`.*

Start Jupyter Notebook to interact with the analysis:

```bash
jupyter notebook

```

*Begin with `notebooks/01_intro_rf_deembedding.ipynb`.*

---

## 4. References and Learning Resources

The mathematical models and physics concepts used in this project are based on industry-standard RF engineering principles.

**Project References & Videos:**

1. [Understanding the Smith Chart (Rohde & Schwarz)](https://www.youtube.com/watch?v=rUDMo7hwihs) - Excellent visual guide for understanding complex impedance mapping.
2. [scikit-rf Documentation](https://scikit-rf.readthedocs.io/en/latest/) - The open-source Python library used for S-parameter manipulations and ABCD matrix conversions.

**Further Academic & Industry Reading (Added for Context):**
3. **"Microwave Engineering" by David M. Pozar** - The absolute gold standard textbook for understanding RF network analysis, S-parameters, and ABCD matrix cascading.
4. **Keysight Technologies Application Note: "De-embedding and Embedding S-Parameter Networks"** - An industry guide explaining the exact mathematical limits and real-world applications of subtracting fixture influences in VNA measurements.

```