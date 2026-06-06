import torch
import torch.nn as nn
import copy
from core.crypto_nets import EncryptorDecoder, Eavesdropper
from core.data_feeder import SecretGenerator, MessageGenerator

class FederatedWorker:
    def __init__(self, worker_id, block_dim, batch_sz, lr, device):
        self.id = worker_id
        self.device = device
        self.sender = EncryptorDecoder(block_dim).to(device)
        self.receiver = EncryptorDecoder(block_dim).to(device)
        self.eaves = Eavesdropper(block_dim).to(device)

        self.opt_pair = torch.optim.Adam(
            list(self.sender.parameters()) + list(self.receiver.parameters()),
            lr=lr
        )
        self.opt_adv = torch.optim.Adam(self.eaves.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.secret_gen = SecretGenerator(block_dim, batch_sz)
        self.plain_gen = MessageGenerator(block_dim, batch_sz)
        self.batch_sz = batch_sz

    def get_weights(self):
        return {
            'sender': copy.deepcopy(self.sender.state_dict()),
            'receiver': copy.deepcopy(self.receiver.state_dict())
        }

    def set_weights(self, state):
        self.sender.load_state_dict(state['sender'])
        self.receiver.load_state_dict(state['receiver'])

    def local_update(self, batches, epochs):
        self.sender.train()
        self.receiver.train()
        self.eaves.train()

        train_turn = 0   # 0: 训练通信对, 1: 训练Eve

        for _ in range(epochs):
            for _ in range(batches):
                # 生成数据
                msg_batch = self.plain_gen.next()
                key_batch = self.secret_gen.next()
                M = torch.tensor(msg_batch, dtype=torch.float32, device=self.device)
                K = torch.tensor(key_batch, dtype=torch.float32, device=self.device)

                self.opt_pair.zero_grad()
                self.opt_adv.zero_grad()

                if train_turn == 0:
                    # 训练通信对（完整对抗损失）
                    C = self.sender(torch.cat([M, K], dim=1))
                    M_rec = self.receiver(torch.cat([C, K], dim=1))
                    M_adv = self.eaves(C)                     # 不detach，允许梯度流向Eve（但不会更新Eve）
                    loss_rec = self.loss_fn(M_rec, M)
                    loss_adv = self.loss_fn(M_adv, M)
                    loss_comm = loss_rec + ((1.0 - loss_adv) ** 2)
                    loss_comm.backward()
                    self.opt_pair.step()
                else:
                    # 训练Eve
                    C = self.sender(torch.cat([M, K], dim=1)).detach()
                    M_adv = self.eaves(C)
                    loss_adv = self.loss_fn(M_adv, M)
                    loss_adv.backward()
                    self.opt_adv.step()

                train_turn = (train_turn + 1) % 2

        return self.get_weights()