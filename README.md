# TemperatureMeasurement

## Overview

This project is to calculate the human body temperature and count the number of entering person with FlIR camera 
connected to Raspberry Pi.

## Structure

- src

    The source code to calculate the human body temperature and count person

- utils

    The source code to manage the folder and files in this project
    
- app

    The main execution file
    
- requirements

    All the dependencies for this project
    
## Installation

- Environment

    Python 3.6

- Dependency Installation

    Please go ahead to the directory of this project and run the following command.
    
    ```
    pip3 install -r requirements.txt
    ```

## Execution

- For the temperature measurement, you can set THERMAL_COEFF_1 and THERMAL_COEFF_2 in settings file, whose default values 
are 0.0439 and -321.

- Please run the following command

    ```
    python3 app.py
    ```
