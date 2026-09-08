"""SAC update rule (Haarnoja et al., 2018): twin-Q critics with Polyak-averaged target
networks, reparameterized policy gradient, automatic temperature (entropy coefficient)
tuning. Matches the standard reference recipe (e.g. CleanRL's `sac_continuous_action.py`).
"""

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SACAgent(object):
    def __init__(self, actor, critic, config, device):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.device = device
        self.cfg = config
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.max_grad_norm = config.get("max_grad_norm", 0.5)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.get("actor_lr", 3e-4))
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.get("critic_lr", 3e-4))

        # Auto temperature tuning (SAC v2, Haarnoja et al. "Soft Actor-Critic Algorithms and
        # Applications"): target entropy defaults to -action_dim, a heuristic that works well
        # across the continuous-control literature and needs no environment-specific tuning.
        target_entropy = config.get("target_entropy")
        self.target_entropy = float(target_entropy) if target_entropy is not None else -2.0

        # `init_alpha` — KHONG phai 1.0 (log_alpha = 0) nhu mac dinh cua SAC tu con so 0.
        # Ly do rat cu the o day: actor_loss = alpha * log_prob - min_Q. Trong nhung update
        # dau, critic con NGAU NHIEN nen min_Q ~ 0, tuc gradient cua actor gan nhu chi con
        # so hang alpha * log_prob = mot lenh "tang entropy" thuan tuy. Voi alpha = 1.0 va
        # log_std_init = -2.5 (std 0.08, do rong nhieu ma warm-start IL can giu), do lech
        # chuan bi day len rat nhanh truoc khi alpha kip tu dieu chinh xuong (alpha_lr
        # 3e-4) — tuc chinh cai warm-start ma toan bo file nay sinh ra de bao ve bi xoa
        # trong vai nghin buoc dau. 0.1 giu ap luc entropy nho gap 10 lan; auto-tuning van
        # keo alpha ve dung cho no can sau do, chi la khong pha gi tren duong di.
        init_alpha = float(config.get("init_alpha", 0.1))
        if init_alpha <= 0.0:
            raise ValueError("init_alpha phai > 0, nhan duoc %r" % (init_alpha,))
        self.log_alpha = torch.tensor([math.log(init_alpha)], requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.get("alpha_lr", 3e-4))

        # Doi xung voi `critic_warmup_updates` cua PPO (../drl_training/ppo/ppo_agent.py): trong N
        # gradient step dau, CHI critic duoc hoc. Khong co no thi actor bat dau di chuyen
        # theo mot ham Q hoan toan ngau nhien — cung mot van de ma PPO da phai xu ly, va o
        # SAC no con nang hon vi actor duoc cap nhat MOI env step chu khong phai moi rollout.
        self.critic_warmup_steps = int(config.get("critic_warmup_steps", 0))
        self.updates_done = 0

        # --- Rang buoc BC (TD3+BC, Fujimoto & Gu 2021) ---------------------------------
        # `bc_coef = 0` tat han -> SAC thuan.
        #
        # Vi sao can: SAC KHONG co vung tin cay. PPO co `clip_range` chan ti so importance
        # sampling va `target_kl` cat epoch som, nen policy khong the roi xa diem khoi dau
        # trong mot update. SAC toi uu `alpha*log_prob - min_Q` tu do, va Adam chuan hoa do
        # lon gradient nen moi tham so dich ~lr moi buoc BAT KE gradient lon hay nho — mot
        # huong sai nhung nhat quan se tich luy khong gi can lai.
        #
        # Do duoc tren runs/sac_v2_smoke2 va smoke3, cung bai 64 quan sat, so voi IL:
        #     actor_lr 3e-5: lenh lai -0.001 -> -0.274 | ga +0.288 -> -0.372
        #     actor_lr 1e-5: lenh lai -0.001 -> +0.203 | ga +0.288 -> +0.639
        # Huong tuy tien, do lon tuong duong, va ha lr 3 lan khong ngan duoc — dung dau hieu
        # cua "khong co gi giu policy lai" chu khong phai "hoc qua nhanh". Ca hai lan critic
        # deu tot truoc khi tha actor (EV 0.95), nen cung khong phai loi critic.
        #
        # Cach chuan cho dung tinh huong nay (warm-start tu policy bat chuoc) la keo policy
        # ve phia hanh vi goc. `bc_actor` la ban sao DONG BANG cua actor ngay sau warm-start.
        # He so lambda chuan hoa theo |Q| dung nhu TD3+BC, de `bc_coef` khong phu thuoc thang
        # reward — dieu quan trong o day vi thang do vua doi 15 lan khi chuyen sang
        # reward "normalized".
        # --- Dong bang `log_std_head` --------------------------------------------------
        # SAC dung log_std PHU THUOC TRANG THAI (mot head rieng) — do la thiet ke chinh
        # thong cua no, va dung cho viec hoc tu con so 0: no bom them tham do vao dung
        # nhung trang thai ma ham Q con bat dinh.
        #
        # O day dieu do phan tac dung. Do tren runs/sac_v2_smoke4 (da co rang buoc BC nen
        # policy trung binh gan nhu khong doi): `log_std_head.weight` phinh tu 0.0010 len
        # 0.0113 (11 lan) chi sau 1000 buoc actor, khien std LAI vot tu 0.0783 len 0.1222
        # (+56%) va std GA tu 0.3092 len 0.4933 (+60%) o mot so trang thai. Tuc xe duoc bom
        # them nhieu lai o dung nhung tinh huong kho — noi no it chiu duoc nhieu nhat. Va
        # cham tang 53% -> 88% du policy trung binh chi troi 0.035.
        #
        # PPO khong dinh loi nay vi `GaussianActor.log_std` la MOT VECTOR doc lap trang
        # thai. Dong bang head nay dua cau truc nhieu cua SAC ve dung nhu PPO: co dinh tai
        # `action_std` cua checkpoint IL. Entropy khi do la hang so, nen alpha tu dong giam
        # ve gan 0 va so hang entropy that su bien mat — day la lua chon CO Y: warm-start
        # tu policy bat chuoc thi tham do nen do nguoi dat, khong de thuat toan tu noi rong.
        self.freeze_log_std = bool(config.get("freeze_log_std", False))
        if self.freeze_log_std:
            for name, param in self.actor.named_parameters():
                if name.startswith("log_std_head"):
                    param.requires_grad = False

        self.bc_coef = float(config.get("bc_coef", 0.0))
        # Thang de CHUAN HOA sai so BC theo tung chieu — mac dinh la `action_std` cua
        # checkpoint IL, (0.078, 0.306).
        #
        # Vi sao bat buoc: `F.mse_loss(mean_action, bc_mean)` lay trung binh tren CA HAI
        # chieu, ma hai chieu chenh nhau 30 lan ve bien do. Do tren 256 quan sat THAT
        # (probe_obs.npz): |lenh lai| dien hinh cua IL la 0.0231 con |lenh ga| la 0.6916.
        # Sai so ga vi the nuot toan bo MSE va rang buoc len chieu LAI gan nhu bien mat —
        # do duoc: runs/sac_v2_smoke4 (bc_coef 2.5, chua chuan hoa) van troi 0.0103 tren
        # chieu lai, tuc 45% chinh bien do tin hieu lai.
        #
        # Day dung la loi thang do da gap o ham reward cua PPO (so hang toc do nuot so hang
        # bam lan 14:1), lap lai o mot cho khac. Chia moi chieu cho do lech chuan cua no
        # truoc khi tinh MSE lam ca hai chieu dong gop tuong duong.
        bc_scale = config.get("bc_action_scale")
        if bc_scale is None:
            bc_scale = [0.0782, 0.3058]
        self.bc_scale = torch.tensor([float(v) for v in bc_scale], device=device).clamp(min=1e-3)
        self.bc_actor = None
        if self.bc_coef > 0:
            self.bc_actor = copy.deepcopy(self.actor).to(device)
            for param in self.bc_actor.parameters():
                param.requires_grad = False
            self.bc_actor.eval()

        # Xem docstring `policy.backbone.freeze_batchnorm`. SAC can dieu nay khong kem PPO:
        # `select_action()` chay tren batch = 1 con `update()` chay tren batch 128, va target
        # critic thi duoc cap nhat bang Polyak — de thong ke BatchNorm troi theo batch size
        # se lam target va online critic lech nhau mot cach khong the truy vet.
        from policy.backbone import freeze_batchnorm
        self.frozen_bn = freeze_batchnorm(self.actor, self.critic)
        if getattr(self, "critic_target", None) is not None:
            self.frozen_bn += freeze_batchnorm(self.critic_target)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(self, seg, scalar, deterministic=False):
        """Single-observation inference — used both by `train_sac.py`'s rollout loop and
        `evaluate.py`. Unlike PPO, SAC doesn't need `log_prob`/value alongside the acting
        call: the update step recomputes both fresh from replay-buffer samples, fully
        decoupled from how the action was originally chosen during collection."""
        seg_t = torch.as_tensor(seg, device=self.device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=self.device).unsqueeze(0)
        action = self.actor.select_action(seg_t, scalar_t, deterministic=deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch, freeze_actor=None):
        """`freeze_actor=None` (mac dinh): tu quyet dinh theo `critic_warmup_steps`.
        `True`/`False` de ep bang tay. Khi dong bang, actor va alpha van duoc TINH de ghi
        log nhung khong co optimizer step nao chay tren chung."""
        if freeze_actor is None:
            freeze_actor = self.updates_done < self.critic_warmup_steps
        self.updates_done += 1

        seg, scalar, action = batch["seg"], batch["scalar"], batch["action"]
        reward, done = batch["reward"], batch["done"]
        next_seg, next_scalar = batch["next_seg"], batch["next_scalar"]

        # --- critic ---
        with torch.no_grad():
            next_action, next_log_prob, _mean = self.actor.sample(next_seg, next_scalar)
            target_q1, target_q2 = self.critic_target(next_seg, next_scalar, next_action)
            target_q = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            # `gamma_n` = gamma^k voi k la SO BUOC that su cong don duoc (xem
            # sac/replay_buffer.py::_rollout). Voi n_step=1 no bang gamma o moi phan tu, nen
            # duong nay trung khit ban 1-step cu. Buffer cu (khong co khoa nay) van chay.
            discount = batch.get("gamma_n")
            if discount is None:
                discount = self.gamma
            y = reward + discount * (1.0 - done) * target_q

        q1, q2 = self.critic(seg, scalar, action)
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        # `explained_variance` = 1 - Var(y - Q1)/Var(y), do tren chinh batch nay.
        #
        # Bat buoc phai co, khong phai de cho dep: `critic_loss` la sai so TUYET DOI nen no
        # phu thuoc thang cua `y`, ma thang do doi theo tung batch. Do tren PPO
        # (runs/ppo_v1), tuong quan giua value_loss va thanh phan episode cua chinh update
        # do la r = -0.89 — tuc value_loss chu yeu phan anh batch nao, khong phai critic tot
        # hay xau. EV chuan hoa theo phuong sai cua muc tieu nen so sanh duoc giua cac buoc.
        #
        # O SAC no con quan trong hon PPO: actor_loss = alpha*log_prob - min_Q duoc toi uu
        # TU DO, khong co vung tin cay nao (PPO co clip_range + target_kl cat epoch som).
        # Neu tha actor khi Q con vo nghia, Adam se tich luy dich chuyen theo mot huong tuy
        # y nhung NHAT QUAN. Do duoc tren runs/sac_v2_smoke2: sau 650 buoc actor, lenh lai
        # xac dinh troi tu -0.001 sang -0.274 (lech trai co dinh) va lenh ga tu +0.288 sang
        # -0.372 (ga thanh phanh) — warm-start IL bi xoa sach. EV la thu cho biet khi nao
        # critic da du tot de tha actor.
        with torch.no_grad():
            y_var = y.var()
            explained_variance = (float("nan") if y_var < 1e-8 else
                                  float(1.0 - (y - q1).var() / y_var))

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
        self.critic_optimizer.step()

        # --- actor (reparameterized policy gradient) ---
        with torch.set_grad_enabled(not freeze_actor):
            new_action, log_prob, _mean = self.actor.sample(seg, scalar)
            q1_pi, q2_pi = self.critic(seg, scalar, new_action)
            min_q_pi = torch.min(q1_pi, q2_pi)
            if self.bc_actor is None:
                actor_loss = (self.alpha.detach() * log_prob - min_q_pi).mean()
                bc_loss = torch.zeros((), device=min_q_pi.device)
            else:
                # lambda = bc_coef / mean|Q| — so hang Q duoc chuan hoa ve do lon 1, nen
                # so hang BC (he so 1) luon so sanh duoc voi no bat ke thang reward.
                lmbda = self.bc_coef / min_q_pi.abs().mean().detach().clamp(min=1e-6)
                _a, _lp, mean_action = self.actor.sample(seg, scalar)
                with torch.no_grad():
                    _b, _blp, bc_mean = self.bc_actor.sample(seg, scalar)
                bc_loss = F.mse_loss(mean_action / self.bc_scale, bc_mean / self.bc_scale)
                actor_loss = (self.alpha.detach() * log_prob).mean()                     - lmbda * min_q_pi.mean() + bc_loss

        if not freeze_actor:
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.actor_optimizer.step()

        # --- temperature ---
        # Cung dong bang trong warmup: alpha chi co y nghia nhu mot muc gia cho entropy cua
        # actor: chinh no trong khi actor dung yen la dieu chinh mot thu khong ai dung toi.
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        if not freeze_actor:
            self.alpha_optimizer.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # --- Polyak-averaged target critic ---
        with torch.no_grad():
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.mul_(1.0 - self.tau).add_(param, alpha=self.tau)

        return {
            "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(), "alpha": float(self.alpha.item()),
            "mean_q": float(min_q_pi.mean().item()), "entropy": float(-log_prob.mean().item()),
            "explained_variance": explained_variance,
            "bc_loss": float(bc_loss.item()),
            "freeze_actor": bool(freeze_actor),
        }

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "updates_done": self.updates_done,
        }

    def load_state_dict(self, state, actor_only=False):
        """`actor_only=True`: chi nap actor, bo qua critic va cac optimizer.

        Danh gia (`evaluate.py`) chi goi `select_action` -> chi dung actor; critic khong
        tham gia mot phep tinh nao. Nen mot checkpoint train truoc khi kien truc `QNetwork`
        doi (vd truoc khi them `action_emb`) VAN danh gia duoc dung, du `critic` trong do
        khong con khop shape. Mac dinh van la False: luc TRAIN thi critic lech shape la loi
        that su, im lang bo qua se lam agent hoc lai tu critic ngau nhien ma khong ai biet.
        """
        self.actor.load_state_dict(state["actor"])
        if actor_only:
            return
        self.critic.load_state_dict(state["critic"])
        if "critic_target" in state:
            self.critic_target.load_state_dict(state["critic_target"])
        else:
            self.critic_target = copy.deepcopy(self.critic)
            for param in self.critic_target.parameters():
                param.requires_grad = False
        if "actor_optimizer" in state:
            self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        if "critic_optimizer" in state:
            self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        if "log_alpha" in state:
            with torch.no_grad():
                self.log_alpha.copy_(state["log_alpha"].to(self.device))
        if "alpha_optimizer" in state:
            self.alpha_optimizer.load_state_dict(state["alpha_optimizer"])
        # Thieu key nay = checkpoint truoc ban co critic warmup -> coi nhu da qua warmup,
        # dung hon la bat actor dong bang lai mot lan nua tren mot critic da hoc xong.
        self.updates_done = int(state.get("updates_done", self.critic_warmup_steps))
