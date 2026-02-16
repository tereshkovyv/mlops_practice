#!/bin/bash

echo "DATA CREATION"
python data_creation.py
echo "MODEL PREPARATION"
python model_preparation.py
echo "MODEL TESTING"
python model_testing.py