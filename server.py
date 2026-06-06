import torch
import os
from settings import defaults
from core.local_worker import FederatedWorker
from core.crypto_nets import EncryptorDecoder, Eavesdropper
from core.data_feeder import SecretGenerator, MessageGenerator

class FedAvgCoordinator:
    def __init__(self):
        # 安全读取配置，如果缺失则使用默认值
        fed_cfg = defaults.get('federated', {})
        model_cfg = defaults.get('basic', {})

        self.num_workers = fed_cfg.get('client_count', 2)
        self.rounds = fed_cfg.get('comm_rounds', 20)
        self.local_epochs = fed_cfg.get('local_epochs', 5)
        self.local_batches = fed_cfg.get('local_batches', 256)

        self.block_dim = model_cfg.get('block_dim', 16)
        self.batch_sz = model_cfg.get('batch_size', 64)
        self.learning_rate = model_cfg.get('lr_comm', 0.001)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        print(f"Config: workers={self.num_workers}, rounds={self.rounds}, "
              f"local_epochs={self.local_epochs}, local_batches={self.local_batches}")

        # 创建客户端
        self.workers = []
        for i in range(self.num_workers):
            w = FederatedWorker(i, self.block_dim, self.batch_sz,
                                self.learning_rate, self.device)
            self.workers.append(w)

        # 全局参数（用第一个客户端的参数结构）
        self.global_weights = self.workers[0].get_weights()

        self.history = {'round': [], 'bob_mse': [], 'eve_mse': []}

    def average_weights(self, weight_list):
        avg = {'sender': {}, 'receiver': {}}
        for key in self.global_weights['sender'].keys():
            sender_stack = torch.stack([w['sender'][key].float() for w in weight_list])
            receiver_stack = torch.stack([w['receiver'][key].float() for w in weight_list])
            avg['sender'][key] = torch.mean(sender_stack, dim=0)
            avg['receiver'][key] = torch.mean(receiver_stack, dim=0)
        return avg

    def test_global_model(self, round_idx):
        sender = EncryptorDecoder(self.block_dim).to(self.device)
        receiver = EncryptorDecoder(self.block_dim).to(self.device)
        eaves = Eavesdropper(self.block_dim).to(self.device)

        sender.load_state_dict(self.global_weights['sender'])
        receiver.load_state_dict(self.global_weights['receiver'])
        sender.eval()
        receiver.eval()
        eaves.eval()

        secret_gen = SecretGenerator(self.block_dim, self.batch_sz)
        plain_gen = MessageGenerator(self.block_dim, self.batch_sz)

        msg = plain_gen.next()
        key = secret_gen.next()
        M = torch.tensor(msg, dtype=torch.float32, device=self.device)
        K = torch.tensor(key, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            C = sender(torch.cat([M, K], dim=1))
            M_rec = receiver(torch.cat([C, K], dim=1))
            M_adv = eaves(C)
            bob_mse = torch.nn.MSELoss()(M_rec, M).item()
            eve_mse = torch.nn.MSELoss()(M_adv, M).item()

        print(f"[Round {round_idx+1:2d}] Decoder MSE: {bob_mse:.6f}, Adversary MSE: {eve_mse:.6f}")
        self.history['round'].append(round_idx+1)
        self.history['bob_mse'].append(bob_mse)
        self.history['eve_mse'].append(eve_mse)

    def run(self):
        print("\n=== Federated Learning with Neural Cryptography ===")
        print(f"Workers: {self.num_workers}, Rounds: {self.rounds}, "
              f"Local updates: {self.local_batches} batches x {self.local_epochs} epochs\n")

        for r in range(self.rounds):
            # 分发全局参数
            for w in self.workers:
                w.set_weights(self.global_weights)

            # 本地训练
            updates = []
            for w in self.workers:
                new_w = w.local_update(batches=self.local_batches, epochs=self.local_epochs)
                updates.append(new_w)

            # 聚合
            self.global_weights = self.average_weights(updates)

            # 评估
            self.test_global_model(r)

        print("\n=== Training finished ===")
        torch.save(self.global_weights, 'models/fed_global.pth')
        print("Global model saved to models/fed_global.pth")

        # 绘制联邦学习收敛曲线
        try:
            import matplotlib.pyplot as plt
            plt.figure()
            plt.plot(self.history['round'], self.history['bob_mse'], label='Decoder MSE')
            plt.plot(self.history['round'], self.history['eve_mse'], label='Adversary MSE')
            plt.xlabel('Round')
            plt.ylabel('MSE')
            plt.title('Federated Neural Cryptography Convergence')
            plt.legend()
            plt.savefig('fed_convergence.png')
            plt.close()
            print("Saved convergence plot to fed_convergence.png")
        except ImportError:
            print("Matplotlib not installed, skip plotting.")

        # 绘制对比柱状图（需要集中式训练的最终MSE文件）
        centralized_mse = None
        if os.path.exists('centralized_final_mse.txt'):
            with open('centralized_final_mse.txt', 'r') as f:
                centralized_mse = float(f.read().strip())
        else:
            print("centralized_final_mse.txt not found, skip comparison plot.")

        if centralized_mse is not None:
            fed_final_mse = self.history['bob_mse'][-1] if self.history['bob_mse'] else 1.0
            try:
                plt.figure()
                models = ['Federated (FedANC)', 'Centralized']
                final_mse = [fed_final_mse, centralized_mse]
                plt.bar(models, final_mse, color=['blue', 'green'])
                plt.ylabel('Decoder MSE (log scale)')
                plt.yscale('log')
                plt.title('Comparison of Final Decryption Error')
                for i, v in enumerate(final_mse):
                    plt.text(i, v + max(final_mse)*0.05, f'{v:.2e}', ha='center')
                plt.savefig('comparison.png')
                plt.close()
                print("Saved comparison plot to comparison.png")
            except Exception as e:
                print(f"Failed to draw comparison plot: {e}")

def launch_federated():
    coordinator = FedAvgCoordinator()
    coordinator.run()

if __name__ == "__main__":
    launch_federated()