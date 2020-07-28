import torch



if __name__ == "__main__":


    x = torch.randint(0,10,(3,5))
    print(x)
    values, indices = torch.max(x, dim=1)
    print(values)
    print(indices)
