# CartPole Simulation with DQN

This project implements a CartPole simulation using Deep Q-Networks (DQN) for reinforcement learning. The project includes two versions of the CartPole game and their respective DQN implementations.

- V1 Finds its balance within the frame
- V2 Find its balance as close to the arrow as possible

## V1 Demo



https://github.com/user-attachments/assets/bd76af18-7f96-4886-8f07-4336cdba0416



## V2 Demo



https://github.com/user-attachments/assets/15dcbac3-d69c-40db-bb71-03018ca53087



## Project Structure

.
├── CartPoleGame.py
├── CartPoleGameV2.py
├── QN.py
├── QNV2.py
├── testV1.py
├── testV2.py
├── model/
│ └── good/
│ ├── cartpole_4391_record_133.pth
│ └── cartpoleV2_4025_record_64.pth

## Files

- `CartPoleGame.py`: Contains the implementation of the first version of the CartPole simulation.
- `CartPoleGameV2.py`: Contains the implementation of the second version of the CartPole simulation with a features where you click the screen and the cart moves to that position.
- `QN.py`: Implements the DQN agent for the first version of the CartPole simulation.
- `QNV2.py`: Implements the DQN agent for the second version of the CartPole simulation.
- `testV1.py`: Script to test the DQN agent on the first version of the CartPole simulation.
- `testV2.py`: Script to test the DQN agent on the second version of the CartPole simulation.
- `model/good/`: Directory containing pre-trained models.

## Requirements

- Python 3.x
- Pygame
- NumPy
- Matplotlib
- PyTorch

You can install the required packages using pip:

`sh pip install pygame numpy matplotlib torch `

## Running the Simulations

### CartPoleGame

To run the first version of the CartPole simulation:
`sh python CartPoleGame.py`

### CartPoleGameV2

To run the second version of the CartPole simulation:
`sh python CartPoleGameV2.py`

## Training the DQN Agent

### For CartPoleGame

To train the DQN agent on the first version of the CartPole simulation:

`sh python QN.py`

### For CartPoleGameV2

To train the DQN agent on the second version of the CartPole simulation:

`sh python QNV2.py`

## Testing the DQN Agent

### For CartPoleGame

To test the DQN agent on the first version of the CartPole simulation:

`sh python testV1.py`

### For CartPoleGameV2

To test the DQN agent on the second version of the CartPole simulation:

`sh python testV2.py`

## Pre-trained Models

Pre-trained models are available in the good directory:

- cartpole_4391_record_133.pth: Pre-trained model for the first version of the CartPole simulation.
- cartpoleV2_4025_record_64.pth: Pre-trained model for the second version of the CartPole simulation.

You can load these models in the respective test scripts to evaluate their performance.

### License

This project is licensed under the MIT License.
