import torch
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
from settings import defaults
from core.crypto_nets import EncryptorDecoder, Eavesdropper
from core.data_feeder import SecretGenerator, MessageGenerator

class CentralizedTrainer:
    def __init__(self, debug=False):
        cfg = defaults["basic"]
        self.block_dim = cfg["block_dim"]
        self.batch_len = cfg["batch_size"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.alice = EncryptorDecoder(self.block_dim).to(self.device)
        self.bob = EncryptorDecoder(self.block_dim).to(self.device)
        self.eve = Eavesdropper(self.block_dim).to(self.device)

        self.criterion = torch.nn.MSELoss()

        self.key_gen = SecretGenerator(self.block_dim, self.batch_len)
        self.plain_gen = MessageGenerator(self.block_dim, self.batch_len)

        self.history_loss = []   # 记录每个 batch 的 Bob 损失

    def train(self, batches, epochs):
        print("Centralized neural cryptography training (full adversarial loss)")
        ab_params = itertools.chain(self.alice.parameters(), self.bob.parameters())
        opt_alice = torch.optim.Adam(ab_params, lr=defaults["basic"]["lr_comm"])
        opt_eve = torch.optim.Adam(self.eve.parameters(), lr=defaults["basic"]["lr_adv"])

        self.alice.train()
        self.bob.train()
        self.eve.train()

        train_turn = 0   # 0: 训练通信对, 1: 训练Eve

        for ep in range(epochs):
            print(f"Epoch {ep+1}/{epochs}")
            for _ in tqdm(range(batches)):
                # 生成数据
                plain = self.plain_gen.next()
                key = self.key_gen.next()
                P = torch.tensor(plain, dtype=torch.float32, device=self.device)
                K = torch.tensor(key, dtype=torch.float32, device=self.device)

                opt_alice.zero_grad()
                opt_eve.zero_grad()

                if train_turn == 0:
                    # 训练通信对（Alice & Bob）
                    C = self.alice(torch.cat([P, K], dim=1))
                    Pb = self.bob(torch.cat([C, K], dim=1))
                    Pe = self.eve(C)
                    loss_bob = self.criterion(Pb, P)
                    loss_eve = self.criterion(Pe, P)
                    loss_alice = loss_bob + ((1.0 - loss_eve) ** 2)
                    loss_alice.backward()
                    opt_alice.step()

                    self.history_loss.append(loss_bob.item())
                else:
                    # 训练Eve
                    C = self.alice(torch.cat([P, K], dim=1)).detach()
                    Pe = self.eve(C)
                    loss_eve = self.criterion(Pe, P)
                    loss_eve.backward()
                    opt_eve.step()

                train_turn = (train_turn + 1) % 2

        # 绘制损失曲线
        plt.figure()
        plt.plot(self.history_loss)
        plt.xlabel('Batch')
        plt.ylabel('Bob Reconstruction MSE')
        plt.title('Centralized Training Loss (Full Adversarial)')
        plt.savefig('centralized_loss.png')
        plt.close()
        print("Saved centralized loss curve to centralized_loss.png")

        final_mse = self.history_loss[-1] if self.history_loss else 1.0
        with open('centralized_final_mse.txt', 'w') as f:
            f.write(f"{final_mse:.10f}")
        print(f"Centralized final MSE = {final_mse:.10f} (saved to centralized_final_mse.txt)")

        # 评估 Eve 的 MSE
        self.alice.eval()
        self.bob.eval()
        self.eve.eval()
        with torch.no_grad():
            test_plain = self.plain_gen.next()
            test_key = self.key_gen.next()
            P_test = torch.tensor(test_plain, dtype=torch.float32, device=self.device)
            K_test = torch.tensor(test_key, dtype=torch.float32, device=self.device)
            C_test = self.alice(torch.cat([P_test, K_test], dim=1))
            Pe_test = self.eve(C_test)
            eve_mse = self.criterion(Pe_test, P_test).item()
            print(f"Centralized Eve MSE = {eve_mse:.6f}")
            with open('centralized_eve_mse.txt', 'w') as f:
                f.write(f"{eve_mse:.6f}")

        print("Training finished")
        return self.alice, self.bob, self.eve

def start_training(model_name="basic"):
    if model_name != "basic":
        print("Only 'basic' model is implemented in standalone mode.")
        return
    trainer = CentralizedTrainer()
    trainer.train(batches=defaults["train_config"]["batch_num"],
                  epochs=defaults["train_config"]["epoch_num"])