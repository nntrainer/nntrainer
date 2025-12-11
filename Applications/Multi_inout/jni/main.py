"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>

@file main.py
@date 12 December 2025
@brief Python script for multi-input multi-output neural network example
@author Seungbaek Hong <sb92.hong@samsung.com>
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import Tuple, List


##
# @brief MultiInoutNet class to verify multi input and multi output
class MultiInoutNet(nn.Module):
    def __init__(self):
        super(MultiInoutNet, self).__init__()
        
        self.shared_lstm = nn.LSTM(input_size=2, hidden_size=2, batch_first=True)
        self.shared_fc = nn.Linear(2, 2)
        
        self.output_1 = nn.Linear(4, 1)
        self.output_2 = nn.Linear(4, 1)
        
        self.relu = nn.ReLU()
        
    ##
    # @brief forward function to run the model
    def forward(self, input0: torch.Tensor, input1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fc_out = self.shared_fc(input0)
        fc_out = self.relu(fc_out)
        
        batch_size = input1.size(0)
        lstm_out, (hidden, cell) = self.shared_lstm(input1)
        lstm_out = lstm_out[:, -1, :]
        
        concat_out = torch.cat([fc_out.squeeze(1), lstm_out], dim=1)
        
        out1 = self.output_1(concat_out)
        out2 = self.output_2(concat_out)
        
        return out1, out2


##
# @brief MultiInoutDataLoader class to load data
class MultiInoutDataLoader:
    def __init__(self, data_size: int = 8):
        self.data_size = data_size
        self.current_idx = 0
        
    ##
    # @brief generate_batch function to generate batch data
    def generate_batch(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        input0 = torch.FloatTensor(batch_size, 1, 2).uniform_(-1.0, 1.0)
        input1 = torch.FloatTensor(batch_size, 4, 2).uniform_(-1.0, 1.0)
        
        first_input_values = input0[:, 0, 0]
        label1 = 2.0 * first_input_values.unsqueeze(1)
        label2 = first_input_values.pow(2).unsqueeze(1)
        
        return input0, input1, label1, label2
    
    ##
    # @brief next function to get next batch data
    def next(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        input0, input1, label1, label2 = self.generate_batch(batch_size)
        
        self.current_idx += batch_size
        last = (self.current_idx >= self.data_size)
        if last:
            self.current_idx = 0
            
        return input0, input1, label1, label2, last


##
# @brief train_model function to train the model
def train_model(model: nn.Module, dataloader: MultiInoutDataLoader, epochs: int = 200, learning_rate: float = 0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Training started...")
    model.train()
    
    for epoch in range(epochs):
        total_loss1 = 0.0
        total_loss2 = 0.0
        batch_count = 0
        
        while True:
            input0, input1, label1, label2, last = dataloader.next(batch_size=32)
            
            optimizer.zero_grad()
            
            output1, output2 = model(input0, input1)
            
            loss1 = criterion(output1, label1)
            loss2 = criterion(output2, label2)
            
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
            
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            batch_count += 1
            
            if last:
                break
        
        if (epoch + 1) % 10 == 0:
            avg_loss1 = total_loss1 / batch_count
            avg_loss2 = total_loss2 / batch_count
            avg_total_loss = avg_loss1 + avg_loss2
            print(f"Epoch {epoch+1}/{epochs} - Loss1: {avg_loss1:.6f}, Loss2: {avg_loss2:.6f}, Total: {avg_total_loss:.6f}")
    
    print("Training completed!")
    return model


##
# @brief run_inference function to run inference
def run_inference(model: nn.Module, dataloader: MultiInoutDataLoader, num_samples: int = 2):
    print("\n=== Running Inference and Printing Samples ===")
    print("Using trained model for inference")
    print(f"\nTesting {num_samples} samples:")
    print("-" * 40)
    
    model.eval()
    
    with torch.no_grad():
        for i in range(2):
            if i == 0:
                input0_val = -0.5
                input0 = torch.tensor([[[-0.5, 0.2]]], dtype=torch.float32)
                
                input1 = torch.tensor([[
                    [-0.9, 0.1],
                    [0.8, -0.2],
                    [0.0, 0.5],
                    [-0.4, 0.7]
                ]], dtype=torch.float32)
            else:
                input0_val = 0.8
                input0 = torch.tensor([[[0.8, -0.9]]], dtype=torch.float32)
                
                input1 = torch.tensor([[
                    [0.3, -0.6],
                    [-0.1, 0.4],
                    [0.9, -0.8],
                    [0.2, -0.5]
                ]], dtype=torch.float32)
            
            label0_val = input0_val * 2.0
            label1_val = input0_val * input0_val
            
            output1, output2 = model(input0, input1)
            
            # Extract and format values for printing
            # input0 is (1, 1, 2), flatten to (2,)
            input0_vals = input0.numpy().flatten() 
            # input1 is (1, 4, 2), flatten to (8,)
            input1_vals = input1.numpy().flatten()
            # output1 and output2 are (1, 1), so .item() extracts the scalar
            pred0_val = output1.item()
            pred1_val = output2.item()
            
            print(f"\nSample {i+1}:")
            # Format Input0 values
            input0_str = ", ".join([f"{x:.6g}" for x in input0_vals])
            print(f"Input0: [{input0_str}]")
            
            # Format Input1 values
            input1_str = ", ".join([f"{x:.6g}" for x in input1_vals])
            print(f"Input1: [{input1_str}]")
            
            # Format Label and Predicted values
            print(f"Label0: {label0_val:.6g}, Predicted0: {pred0_val:.6g}")
            print(f"Label1: {label1_val:.6g}, Predicted1: {pred1_val:.6g}")
    
    print("\n" + "-" * 40)
    print("Inference completed!")


##
# @brief main function
def main():
    print("PyTorch Multi-input-output Neural Network")
    print("=" * 50)
    
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    model = MultiInoutNet()
    dataloader = MultiInoutDataLoader(data_size=1024)
    
    print("\nModel Architecture:")
    print(model)
    print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
    
    trained_model = train_model(model, dataloader, epochs=100, learning_rate=0.01)
    
    run_inference(trained_model, dataloader, num_samples=2)


if __name__ == "__main__":
    main()
