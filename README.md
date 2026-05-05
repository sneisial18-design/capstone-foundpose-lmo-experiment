# Capstone FoundPose LM-O Experiment

This repository contains my experiment scripts, configuration notes, and analysis for reproducing FoundPose on the LM-O dataset.

## Project Goal

The goal of this project is to understand and reproduce a training-free 6DoF object pose estimation pipeline using FoundPose.

The experiment focuses on:

- LM-O dataset preparation
- Offline onboarding
- Template rendering
- Object representation generation
- Online inference
- Result interpretation

## Original Repository

The original FoundPose implementation is available at:

https://github.com/facebookresearch/foundpose

This repository does not redistribute the original FoundPose source code, BOP datasets, 3D object models, or large generated files.

## Repository Structure

- scripts/: Shell scripts for running each stage
- configs_modified/: Modified or documented configuration files
- notes/: Code review and pipeline notes
- figures/: Report and presentation figures
- results/: Small summary results only

## Pipeline Summary

LM-O dataset preparation  
-> Offline onboarding  
-> Template rendering  
-> Object representation generation  
-> Online inference  
-> Evaluation / qualitative analysis

## Notes

Large files such as datasets, templates, representations, and inference outputs are excluded from this repository.
