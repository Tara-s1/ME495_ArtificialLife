# ME495: Artificial Life

## Final Project: Optimizing Leg Geometry of Creatures in Difftaichi

This project explores how a creature can optimize the geometry of its legs to maximize forward speed (x-distance traveled in a given time) while minimizing mass or the number of particles.  

## Approach  

The code uses an evolutionary algorithm, where in each iteration:  
1. Take leading robot design 
2. Perform random mutation 
3. Evaluate performance through a loss function based on forward speed & number of particles
4. Compare loss to current “winning design” and replace if better 
 

Possible mutations for each leg include:  
- Increasing leg length  
- Changing leg structure (hollow to solid or vice versa)  
- Shifting the x-position of the legs left or right  

## How to Use  

1. **Set Initial Conditions:**  
   - Input your desired initial robot in `main`.  
   - Choose a predefined configuration, such as `robot_eight_uniform()`.  
   - Alternatively, generate a random robot with an unknown number of limbs using `robot(scene)`.  

2. **Adjust Parameters:**  
   - Modify the number of mutation iterations.  
   - Set the number of iterations for visualization.  

## Dependencies  

This project relies on the difftaichi package. Installation instructions can be found here:  
[https://github.com/taichi-dev/difftaichi/tree/master](https://github.com/taichi-dev/difftaichi/tree/master)  

## Running the Code  

Navigate to the project repository and run:  

```bash
python3 final_project.py


