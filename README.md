# BlackMatter

### Overview
A family of chess engines powered by lucidrains' Denoising Diffusion Probabilistic Model (https://github.com/lucidrains/denoising-diffusion-pytorch) in Python using torch.
Paper: [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

### Write-up
#### Core Architecture: 
The engine's logic is encapsulated in several Python classes:

- ChessPositionEncoder: This class translates the complex state of a chess board into a numerical format (a tensor) that the neural network can understand. It also decodes the model's output back into a legal chess move.

- DataCollector: Responsible for generating training data by playing games against the installed Stockfish engine. The model learns by observing the moves played by Stockfish.

- ReplayBuffer: Stores the games (position-move pairs) collected by the DataCollector to be used for training the model.

- DDPMChessModel: It uses a Unet1D architecture within a GaussianDiffusion1D framework to learn the patterns of chess moves. It is designed to generate a probability distribution over all possible moves.

- Pipeline: A master class that orchestrates the entire workflow, including data collection, model training, evaluation, and checkpoint management.

- UCIInterface: Implements the Universal Chess Interface (UCI) protocol, enabling the engine to be used with standard chess graphical user interfaces (GUIs)
