# Pygame | 3D Viewer

## Description

Pygame 3D Engine is a Python application developed for simple and efficient 3D model visualization without the need for complex libraries such as OpenGL or GLM. The project uses the Pygame library to render 3D shapes and simulate visual perspectives, making it a lightweight and accessible solution for graphic prototyping and experimentation with 3D computer graphics concepts.  

This project is ideal for developers who want to explore 3D object visualization in a didactic and customizable way.  

### Features  

- Basic rendering of 3D primitives.  
- Simulation of object rotation, scaling, and movement in 3D space.  
- Interactive control of camera and objects.  
- Modular structure for easy expansion.  

## Conda Snippets

### Create environment

``` conda env create -n viwer-env -f ./env.yml ```

### Update environment

``` conda env update -n viwer-env -f ./env.yml ```

### Remove environment

``` conda env remove --n viwer-env ```

### Activate environment

``` conda activate viwer-env ```

### Deactivate environment

``` conda deactivate ```

## Notebooks

``` notebooks/main.ipynb ```

## RUN

``` python src/main.py ```

``` python src/main.py --width 1600 --height 900 --render_type "wireframe" --model_name "./assets/models/box3.obj" ```

``` python src/main.py --width 1600 --height 900 --render_type "solid|shader" --model_name "./assets/models/box3.obj" ```

``` python src/main.py --width 1600 --height 900 --render_type "textured" --model_name "./assets/models/box3.obj" --texture_name "./assets/textures/gold.png" ```


## Links

[github_author](https://github.com/Diegoomal)
