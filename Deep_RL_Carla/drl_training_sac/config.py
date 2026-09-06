"""Shared CLI + JSON config loading for `train_sac.py` / `evaluate.py`.

Mirrors the `--config` JSON-overrides-defaults pattern used by
`data_collection/carla_collector/config.py`, flattened into one dict since the env/agent
code here just does plain keyword lookups (`cfg.get("w_lane_offset", 1.0)`) instead of
argparse groups.

`ENV_DEFAULTS` covers everything the environment needs (CARLA connection, camera, env
dynamics, reward weights, warm-start/IL-checkpoint settings); `SAC_DEFAULTS` adds the
off-policy hyperparameters on top, so `sac_config.json` only needs to *override* what is
actually different — xem chinh file do.

Thu muc nay CHI chay SAC (off-policy). Ban PPO (on-policy) nam o `../drl_training/`; moi
tham chieu toi `train_ppo.py` / `runs/ppo_v*` trong cac chu thich duoi day la BANG CHUNG DO
DUOC ben do, khong phai code trong thu muc nay.
"""

import argparse
import json
from pathlib import Path

ENV_DEFAULTS = {
    # connection
    "host": "127.0.0.1", "port": 2000, "timeout": 20.0,
    # Timeout RIENG cho `client.load_world()` (xem CarlaLaneKeepEnv._load_town). Tach khoi
    # `timeout` vi hai viec khac han nhau ve do lon: mot lenh dieu khien tra loi trong vai
    # chuc mili giay, con nap mot ban do lon mat hang phut.
    "map_load_timeout": 300.0,
    # camera / env
    # width/height  = do phan giai CAMERA segmentation (khop collector: 480x384).
    # obs_width/obs_height = do phan giai OBSERVATION dua vao mang, sau khi
    #   resize_class_map() ha mau. Phai khop IMAGE_WIDTH/IMAGE_HEIGHT cua
    #   train_il_v9.ipynb (240x192) thi actor warm-start moi nhin thay dung thang do
    #   dac trung nhu luc train IL. Ha o day cung giam 4x bo nho rollout/replay
    #   (SAC 50k transition: 9.2GB o 480x384 -> 2.2GB o 240x192).
    "width": 480, "height": 384, "obs_width": 240, "obs_height": 192,
    "fov": 90.0, "fps": 20.0,
    "camera_x": 1.5, "camera_y": 0.0, "camera_z": 2.4, "camera_pitch": -5.0,
    "vehicle_filter": "vehicle.lincoln.mkz2017",
    # fps 20 (0.05s/tick) x action_repeat 4 = 0.2s moi QUYET DINH cua policy = dung
    # `control_dt` ma checkpoint IL da train (5 FPS). Khong the thay bang fps=5 truc tiep:
    # CARLA khuyen cao fixed_delta_seconds <= 0.05s, tren muc do vat ly bat dau sai (xe
    # rung, va cham gia). CarlaLaneKeepEnv._check_control_rate() canh bao neu hai ben lech.
    "action_repeat": 4,
    # Tinh theo QUYET DINH (khong phai tick): 500 x 0.2s = 100s moi episode.
    "max_episode_steps": 500,
    # 10 quyet dinh = 2s lien tuc ngoai lan thi ket thuc episode (truoc day 20 tick = 1s).
    "off_lane_patience_steps": 10,
    # frame_timeout 20s, khong phai 5s. `_get_seg_frame` nem RuntimeError va GIET ca lan
    # chay neu CARLA khong giao anh camera trong khoang nay. 5s qua ngan: do duoc hai lan
    # trong phien nay — runs/ppo_v3 chet o update 198, va eval SAC tren Town04 chet sau 4
    # episode — deu voi cung thong bao "Khong nhan duoc frame camera ... trong 5.0s", trong
    # khi server van song binh thuong ngay sau do. CARLA thinh thoang khung vai giay (nap
    # ban do, GC, tai nen may); 20s chiu duoc nhung khung do ma van du ngan de mot server
    # treo THAT SU khong bien thanh mot lan chay treo im lang.
    "warmup_ticks": 4, "frame_timeout": 20.0, "no_rendering": False, "seed": 42,
    # Ban do nap truoc khi train/eval. None = dung the gioi dang chay tren server.
    # Do tren ca 5 town (ti le waypoint nam trong nga tu — noi `lane_offset_m` va
    # `heading_error_rad` la phep do RAC, xem envs/carla_lane_keep_env.py):
    #     Town01 23.3% | Town04 25.4% | Town05 27.7% | Town02 28.1% | Town03 44.2%
    # Town03 bo di gan mot nua so mau cho bai toan bam lan — dung Town01 (don gian) hoac
    # Town04 (33.8 km lan, nhieu duong dai) de train, de danh Town05 cho danh gia vi do la
    # town ma IL chua tung thay.
    # MAC DINH LA CA BON TOWN MA IL DA TRAIN, khong phai mot town.
    # `best_il_model.pth` ghi `train_towns = ['Town01','Town02','Town03','Town04']` va
    # `val_towns = ['Town05']`. Fine-tune PPO chi tren MOT town lam policy quen ba town kia:
    # do duoc tren runs/ppo_v3 (250 update chi Town04), lech lan tren duong thang di tu
    # 0.098 m (IL) len 0.233 m (Town04, chinh ban do da train) va 0.382 m (Town01) — te hon
    # IL co y nghia thong ke o CA BA ban do (Welch t = 2.22 / 4.38 / 2.55).
    # Giu Town05 lam tap held-out DUY NHAT, dung nhu IL da lam.
    "town": ["Town01", "Town02", "Town03", "Town04"],
    # Doi ban do sau moi bao nhieu episode. `load_world()` mat vai phut nen khong the doi
    # moi episode: 25 episode ~ 8-10 phut lai xe cho ~2 phut nap ban do (~20% chi phi).
    "town_rotate_episodes": 25,
    # Keo camera cua so CARLA bam theo xe ego moi quyet dinh. Chi de NGUOI xem — khong doi
    # observation, khong doi reward, khong doi ket qua train. Tat mac dinh vi trong mot lan
    # train dai thi khong ai ngoi nhin, va no ton mot lenh RPC set_transform moi step.
    "spectator_follow": False,
    # Ghim vong lap ve THOI GIAN THUC. Mac dinh False: o sync mode, CARLA chay nhanh het
    # muc client keo duoc — do duoc 58 tick/s x 0.05s = nhanh gap 2.9 lan thuc te, nen
    # chuyen dong nhin bi "tua nhanh". Bat co nay chi khi NGUOI dang xem; no lam cham
    # training xuong dung toc do that.
    "realtime": False,
    # ------------------------------------------------------------------- reward
    # "normalized": moi so hang chia cho thang tu nhien cua no -> khong thu nguyen, [0,1].
    # "raw"       : cong thuc cu (speed tinh bang m/s tho) — CHI de tai lap runs/ppo_v1..v3.
    #
    # Ban "raw" co mot loi thang do khong sua duoc bang trong so: `w_speed * forward_speed_mps`
    # phu thuoc GIOI HAN TOC DO cua doan duong. Do tren Town04 (co cao toc 90 km/h) so hang
    # nay dat 13.47/buoc trong khi phat lech lan toi da la 1.75 -> ti le 14:1, va cung mot
    # hanh vi lai duoc thuong khac nhau 3 lan giua pho va cao toc. Ban "normalized" chia toc
    # do cho gioi han toc do, lech lan cho nua be rong lan, goc lech cho 45 do; sau do `w_*`
    # moi that su la trong so tuong doi va reward nhat quan giua cac town.
    "reward_mode": "normalized",
    # Toc do (m/s) duoc coi la "day du diem". None = lay trung binh `speed_mps` trong
    # norm_stats cua checkpoint IL (do duoc: 7.56 m/s = 27 km/h) — tuc chinh toc do ma
    # he thong da hoc lai. KHONG dung gioi han toc do hop phap lam mau so: doan cao toc
    # Town04 la 90 km/h, chia cho no thi lai dung nhu IL chi duoc 0.30 diem va reward tong
    # thanh AM, khien "dung yen" tro nen toi uu.
    "target_speed_mps": None,
    # Trong so cho thang [0,1]. Toc do va bam lan gio NGANG NHAU (1.0 : 1.0) thay vi 14:1.
    "w_speed": 1.0, "w_lane_offset": 1.0, "w_heading": 0.4,
    # Hai so hang muot phat HIEU giua hai hanh dong LIEN TIEP, ma hai hanh dong lien tiep
    # khac nhau chinh bang NHIEU THAM DO. Voi hai lan lay mau doc lap tu N(mu, sigma):
    #     E[(a_t - a_{t-1})^2] = 2*sigma^2
    # `log_std` lay tu checkpoint IL cho sigma = (0.078, 0.306), tuc 0.0122 va 0.187.
    # Do la chi phi policy KHONG THE tranh bang cach lai tot hon — chi bang cach thu nho
    # log_std. Nen he so qua lon o day khong day policy lai muot hon, no day policy SUP
    # ENTROPY.
    #
    # Lan dau to dat 20.0 / 2.0 (suy tu "delta 0.05 thi phat 0.05") va do duoc hau qua:
    # 20.0*0.0122 + 2.0*0.187 = 0.62 moi buoc, tren thang ma thuong toc do toi da chi la
    # 1.00 — tuc 62% ngan sach reward bi dot cho nhieu, va mean_ep_reward am o moi update.
    # O ban "raw" cung cong thuc nay chi ton ~1% vi nen reward la 8-25 chu khong phai 1.
    #
    # Gia tri duoi giu chi phi nhieu ~3.7% ma mot cu giat THAT (delta steer 0.3) van ton
    # 1.5*0.09 = 0.135, tuc 14% mot buoc lai het toc do.
    #   w_steer_delta 1.5 -> nhieu 0.018 | giat 0.3 -> 0.135
    #   w_long_delta  0.1 -> nhieu 0.019 | giat 0.5 -> 0.025
    # `w_yaw_rate` cung ha: yaw_rate la dai luong VAT LY (khong phai nhieu hanh dong), va
    # 0.5 se phat 0.125 cho mot cu re binh thuong 0.5 rad/s — tuc phat xe vi da vao cua.
    "w_steer_delta": 1.5, "w_long_delta": 0.1, "w_yaw_rate": 0.1,
    "off_lane_penalty": 2.0,
    # 50.0 (ban cu) bi so hang toc do nuot: `w_speed * forward_speed_mps` cho toi +8.3 moi
    # buoc, nen dam xe chi ton bang 6 buoc chay. Do tren runs/ppo_lane_keep: 14/19 episode
    # ket thuc bang va cham ma van duoc 1000-3600 diem — tin hieu hoc duoc la "cu dam, mien
    # la chay nhanh truoc do". 300 = ~40 buoc = 8 giay lai xe, du de va cham thanh mot su
    # kien dat do that su. Van la mot lua chon can chinh tay: neu xe tro nen qua rut re
    # (dung im de khong bao gio dam) thi ha xuong, hoac tang `w_speed`.
    # 75 = 75 buoc chay het toc do ~ 15 giay lai xe. Giu dung TI LE ma `collision_penalty`
    # 1000 da dat duoc o thang "raw" (1000 / 13.47 = 74 buoc), vi ti le do da duoc kiem
    # chung: va cham luc train giam 48.8% -> 32.6% (z = 2.88) khi doi tu 300 sang 1000.
    "collision_penalty": 75.0, "lane_invasion_penalty": 0.5,
    # Trong nga tu, `lane_offset_m`/`heading_error_rad`/`off_lane` la phep do RAC (waypoint
    # tham chieu nhay sang nhanh khac). Mac dinh tat cac so hang do o day; dat False neu
    # muon chay doi chung voi hanh vi cu. Xem envs/carla_lane_keep_env.py::_compute_reward.
    "junction_mask_lane_terms": True,
    # Trong nga tu, chi coi la ra khoi lan khi lech qua factor x nua be rong lan (~5.2 m).
    # Xem CarlaLaneKeepEnv._compute_reward: dong bang hoan toan bo dem o nga tu tao ra mot
    # lo hong khien xe co the troi han ra khoi mat duong ma khong bi dung — do duoc la
    # nguyen nhan cua 3/3 va cham tren Town04 o runs/ppo_v4.
    "junction_off_lane_factor": 3.0,
    # shared training settings
    "il_checkpoint": "../behavior_cloning/best_il_model.pth",
    "warm_start": True, "device": "cuda", "gamma": 0.99,
}

SAC_DEFAULTS = {
    "output": "./runs/sac_lane_keep",
    # actor_lr THAP hon critic_lr 10 lan, cung ly do va cung ti le nhu PPO
    # (xem ../drl_training/ppo/ppo_agent.py.__init__): actor da warm-start tu IL, critic khoi
    # tao ngau nhien. Cho ca hai cung mot LR nghia la xoa trong so IL bang gradient sinh
    # ra tu mot ham gia tri chua hoc duoc gi.
    #
    # 3e-4 (ban dau, lay tu mac dinh SAC sach) la QUA CAO o day: SAC lam
    # 60000/train_freq 4 = 15 000 gradient step, tuong duong 195 update x 80 minibatch =
    # 15 600 cua PPO. Cung so buoc cap nhat ma LR cao gap 15 lan PPO (2e-5) nghia la
    # actor dich chuyen gap 15 lan — dung co che da pha hong warm-start o runs/ppo_v2/v3.
    # 1e-5, thap hon ca PPO (2e-5), va co ly do cau truc chu khong phai tuy tien:
    # PPO co VUNG TIN CAY — `clip_range` 0.2 chan ti so importance sampling va
    # `target_kl` cat epoch som — nen policy khong the nhay xa trong mot update du
    # gradient lon co nao. SAC khong co gi tuong duong: `actor_loss = alpha*log_prob -
    # min_Q` duoc toi uu tu do. Cong them Adam chuan hoa do lon gradient (moi tham so
    # dich ~lr moi buoc bat ke gradient nho hay lon), nen mot huong gradient NHAT QUAN
    # tu mot critic con vo nghia se tich luy rat nhanh.
    # Do duoc tren runs/sac_v2_smoke2 voi actor_lr 3e-5: chi 650 buoc actor da lam lenh
    # lai xac dinh troi tu -0.001 sang -0.274 va lenh ga tu +0.288 sang -0.372.
    "actor_lr": 1e-5, "critic_lr": 3e-4, "alpha_lr": 3e-4,
    "tau": 0.005,
    # gamma 0.95 RIENG cho SAC, ghi de gia tri 0.99 o ENV_DEFAULTS.
    #
    # Ly do la ti le giua mot hanh dong va chan troi. Moi hanh dong keo dai 0.2s
    # (`action_repeat` 4). Voi gamma 0.99, chan troi hieu dung ~1/(1-g) = 100 buoc =
    # 20 giay, nen MOT hanh dong chiem 1% chan troi — va policy con tu sua o cac buoc
    # sau, nen hau qua dai han cua viec be lai mot nhip that su rat nho. Do la ly do
    # `dQ/da` nho: khong phai critic hoc kem ma la tinh chat cua MDP nay.
    #
    # gamma 0.95 -> chan troi ~20 buoc = 4 giay. Mot hanh dong chiem 5% thay vi 1%, tuc
    # tin hieu tang 5 lan. 4 giay van du dai cho bam lan: hau qua cua mot lenh lai the
    # hien trong 1-3 giay. PPO khong can dieu nay vi GAE CONG DON hang chuc hanh dong
    # doc quy dao thanh mot advantage du lon; SAC phai doc tin hieu tu MOT hanh dong.
    "gamma": 0.95,
    # CANH BAO (do duoc tren runs/sac_f, 6/9/2026): khi `freeze_log_std` = True thi
    # `target_entropy` va `alpha_lr` la HAI NUT CHET. log_std bi dong bang nen policy khong
    # the ha entropy xuong muc target; entropy thuc te dao quanh -2.6 trong khi target la
    # -4.0, nen `alpha_loss` am lien tuc va alpha giam don dieu ve 0:
    #     step  5000 -> 0.0100 | 20000 -> 0.0086 | 30000 -> 0.0039 | 35000 -> 0.0027
    # O step 35000, alpha*log_prob ~ 0.007 tren thang min_Q ~ 21, tuc so hang entropy da tat
    # han (0.03%). Nghia la voi freeze_log_std=True, thuat toan dang chay KHONG con la SAC
    # chuan — no gan voi TD3+BC (twin-Q + actor Gaussian nhieu co dinh) hon. Khong gay hai
    # (alpha di XUONG la chieu an toan; chieu nguoc lai chinh la thu da pha runs/sac_a),
    # nhung phai noi dung nhu vay khi bao cao, va dung ky vong vao viec chinh hai nut nay.
    # Muon auto-temperature that su hoat dong thi phai dat `freeze_log_std` = False.
    #
    # -action_dim = -2.0 (Haarnoja et al.) la mac dinh cho tac vu dieu khien tong quat.
    # O day no qua CAO: entropy muc do dat buoc alpha giu do lech lon tren CHIEU STEER,
    # trong khi lenh lai dien hinh chi 0.005-0.03. Ha xuong -4.0 cho phep policy nhon
    # hon ma van con tham do o chieu longitudinal.
    "target_entropy": -4.0,
    # Nhu PPO: None = lay tu `action_std` cua checkpoint IL (lay trung binh log cua hai
    # chieu, vi SAC dung mot `log_std_head` phu thuoc trang thai chu khong phai vector).
    "log_std_init_from_checkpoint": True,
    # alpha khoi tao (xem sac/sac_agent.py.__init__). `actor_loss = alpha*log_prob -
    # min_Q`, nen gia tri dung phu thuoc THANG CUA Q, va thang do vua doi 15 lan khi
    # reward chuyen sang "normalized":
    #     raw (cao toc Town04) reward/buoc 13.47 -> V ~ 1338
    #     normalized           reward/buoc  0.90 -> V ~   89
    # Q nho di 15 lan thi cung mot alpha co anh huong tuong doi lon gap 15 lan. 0.1 duoc
    # chon cho thang raw; giu nguyen o thang moi se bien nhung update dau thanh mot lenh
    # "tang entropy" ap dao — dung the that bai ma 1.0 da gay ra truoc do, chi la o muc
    # nhe hon. 0.007 giu dung ti le cu; lam tron 0.01.
    "init_alpha": 0.01,
    # He so rang buoc BC (TD3+BC). 0 = tat, SAC thuan. Xem sac/sac_agent.py.__init__:
    # SAC thieu vung tin cay ma PPO co, nen warm-start tu policy IL bi xoa trong vai
    # tram gradient step du actor_lr da ha xuong 1e-5. 2.5 la gia tri TD3+BC dung trong
    # bai goc; vi lambda duoc chuan hoa theo |Q| nen no khong phu thuoc thang reward.
    # Chuan hoa sai so BC theo tung chieu (xem sac/sac_agent.py.__init__). None = dung
    # `action_std` cua checkpoint IL. Khong co no, rang buoc len chieu LAI bi sai so GA
    # nuot mat vi hai chieu chenh 30 lan ve bien do.
    "bc_action_scale": None,
    "bc_coef": 2.5,
    # Dong bang `log_std_head` -> nhieu tham do co dinh tai `action_std` cua IL, doc lap
    # trang thai, dung nhu PPO. Xem sac/sac_agent.py.__init__ de biet so do dan toi lua
    # chon nay. Dat False de quay ve SAC chuan (log_std hoc duoc, phu thuoc trang thai).
    # False: smoke5 do duoc dong bang log_std lam troi chieu LAI xau di (0.0103 ->
    # 0.0275 tren quan sat that) va va cham tang 88% -> 93% so voi smoke4.
    "freeze_log_std": False,
    # So GRADIENT STEP dau CHI train critic (actor + alpha dong bang) — doi xung voi
    # `critic_warmup_updates` cua PPO. Voi train_freq=4, 2000 gradient step = 8000 env
    # step (~6.7 phut mo phong), tuc 13% ngan sach 60k. PPO danh 10/195 update cho viec
    # nay. Dat ve 0 neu muon doi chung khong co warmup.
    "critic_warmup_steps": 2000,
    # Ti le buoc lay hanh dong NGAU NHIEN TOAN DAI trong giai doan critic-warmup, de
    # critic co du lieu hoc su phu thuoc hanh dong cua Q. Chi hoat dong khi actor con
    # dong bang nen khong the pha warm-start. Xem chu thich trong train_sac.py.
    "explore_epsilon": 0.25,
    "log_std_init": -2.5,
    # batch_size / train_freq / total_steps duoc chon theo DO DUOC tren GTX 1650 Max-Q
    # 4 GB (may dang dung), khong phai theo mac dinh sach vo. Mot gradient step SAC
    # (actor sample + 2 critic + 2 target critic + 1 luot critic nua cho actor loss,
    # tren anh one-hot 4 lop 240x192) do duoc:
    #     batch 128 -> 3.920 s, VRAM dinh 2.36 GB
    #     batch  64 -> 1.969 s, VRAM dinh 1.19 GB
    #     batch  32 -> 0.980 s, VRAM dinh 0.60 GB
    # (Do lai cuoi phien; lan do dau phien nhanh gap 2.4 lan — 1.614/0.815/0.411 s — khi
    # may con it tai nen. GPU khong throttle o ca hai lan, xung van 1740/1740 MHz; khac
    # biet den tu cac tien trinh nen tranh GPU. Lay so CHAM lam co so de uoc tinh.)
    # batch 32 + train_freq 4 xu ly cung so mau nhu batch 64 + train_freq 8, nhung gap
    # DOI so buoc toi uu hoa. Voi SAC, nhieu buoc gradient nhieu hon thuong tot hon it
    # buoc muot hon. 0.60 GB VRAM cung an toan hon nhieu canh CARLA tren card 4 GB.
    # Voi train_freq=1 (mac dinh SAC chuan, 1 gradient step moi env step), vong lap bi
    # GHIM o 0.62 step/s — khong phai toc do CARLA (~20 quyet dinh/s) ma la toc do
    # gradient. 100k step se mat 45 GIO, va 2.36 GB VRAM canh CARLA server tren card
    # 4 GB thi gan nhu chac chan OOM.
    #     batch 32 + train_freq 4 + 60k step:
    #         15 000 gradient step x 0.980 s = 4.1 h
    #         + 60 000 / 20 quyet dinh/s     = 0.8 h  -> ~4.9 h, cong ~20% nap ban do
    #         khi xoay 4 town -> ~5.9 h. VRAM 0.60 GB.
    # Danh doi: UTD (update-to-data) = 0.25 thay vi 1.0, tuc moi mau duoc hoc it hon.
    # Chap nhan duoc o day vi actor da warm-start tu IL, khong phai hoc lai tu con so 0.
    # Neu doi sang GPU khoe hon: ha train_freq ve 1-2 va tang total_steps.
    # n-step return. 1 = SAC chuan (1-step TD). Xem sac/replay_buffer.py de biet ly do va
    # danh doi; tom tat: do nhay hanh dong cua critic bao hoa o 8-10% tu step 25 000 tren ca
    # runs/sac_d lan runs/sac_f, va actor chi hoc qua dQ/da nen do la tran cua thuat toan.
    # n-step gop n phan thuong THUC TE vao muc tieu — dung co che GAE cua PPO — nen tin hieu
    # ve hanh dong dau tien manh len ma chan troi hieu dung khong ngan lai.
    "n_step": 1,
    "batch_size": 32, "buffer_capacity": 50000,
    "learning_starts": 2000, "train_freq": 4, "gradient_steps": 1,
    "max_grad_norm": 0.5,
    # 60k quyet dinh x 0.2s = 3.3 gio mo phong.
    "total_steps": 60000, "save_every_steps": 5000,
    "eval_every_steps": 10000, "eval_episodes": 30,
}


def _flatten(data, output=None):
    output = {} if output is None else output
    for key, value in data.items():
        if isinstance(value, dict):
            _flatten(value, output)
        else:
            output[key] = value
    return output


def load_config(argv=None):
    parser = argparse.ArgumentParser(
        description="SAC (off-policy) fine-tuning cho lane-keeping tren CARLA")
    parser.add_argument("--config", default=None,
                        help="File JSON cau hinh, vd sac_config.json")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--il-checkpoint", default=None, help="Checkpoint IL (.pth) de warm-start actor")
    parser.add_argument("--output", default=None, help="Thu muc luu checkpoint + log CSV")
    parser.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                         help="Ghim mo phong ve toc do thuc de xem cho de nhin. Lam CHAM "
                              "training — chi dung khi dang ngoi xem")
    parser.add_argument("--spectator", dest="spectator_follow", action="store_true",
                         default=None,
                         help="Keo camera cua so CARLA bam theo xe de xem truc tiep. Chi anh "
                              "huong hinh anh, khong doi gi ve train")
    parser.add_argument("--town", default=None, nargs="+",
                         help="Mot HOAC NHIEU ban do (vd --town Town01 Town02 Town03 Town04). "
                              "Nhieu ban do = xoay vong moi `town_rotate_episodes` episode, "
                              "bat buoc de khong quen cac town ma IL da hoc. Mot ten = chay "
                              "co dinh (dung cho evaluate.py). Bo qua = dung mac dinh trong "
                              "config")
    parser.add_argument("--resume", default=None, help="Checkpoint DRL (.pt) de resume train / dung de eval")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false", default=None,
                         help="Bo qua warm-start IL — actor khoi tao ngau nhien (chi de doi chung)")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--action-repeat", type=int, default=None,
                         help="So tick vat ly moi quyet dinh cua policy. fps/action_repeat "
                              "phai bang 1/control_dt cua checkpoint IL (mac dinh 20/4 = 5 Hz)")
    parser.add_argument("--max-episode-steps", type=int, default=None,
                         help="So QUYET DINH toi da moi episode (khong phai so tick)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed cho RNG chon diem spawn cua env. Doi gia tri nay giua "
                              "cac phien train noi tiep nhau, neu khong moi phien se gap y "
                              "het mot chuoi kich ban")
    parser.add_argument("--buffer-capacity", type=int, default=None,
                         help="[train_sac.py] so transition toi da trong replay buffer — "
                              "giam gia tri nay truoc tien neu thieu RAM (xem README.md)")
    parser.add_argument("--width", type=int, default=None,
                        help="Do rong CAMERA segmentation (px) — khong phai observation")
    parser.add_argument("--height", type=int, default=None,
                        help="Do cao CAMERA segmentation (px) — khong phai observation")
    parser.add_argument("--obs-width", type=int, default=None,
                        help="Do rong OBSERVATION dua vao mang (mac dinh 240 = khop IL)")
    parser.add_argument("--obs-height", type=int, default=None,
                        help="Do cao OBSERVATION dua vao mang (mac dinh 192 = khop IL)")
    parser.add_argument("--batch-size", type=int, default=None,
                         help="Kich thuoc minibatch update — dat truc tiep VRAM can dung "
                              "(anh one-hot 4 lop o 240x192 nang hon nhieu 160x128, giam "
                              "gia tri nay truoc tien neu OOM, xem README.md)")
    parser.add_argument("--device", default=None, choices=["cuda", "cpu"])
    parser.add_argument("--episodes", type=int, default=None, help="[evaluate.py] so episode danh gia")
    parser.add_argument("--deterministic", action="store_true",
                         help="[evaluate.py] dung mean action thay vi sample (tat exploration)")
    parser.add_argument("--eval-csv-out", default=None,
                         help="[evaluate.py] neu dat, ghi ket qua tung episode ra file CSV nay "
                              "(vd runs/sac_f/eval_Town05.csv)")
    args = parser.parse_args(argv)

    config = dict(ENV_DEFAULTS)
    config.update(SAC_DEFAULTS)
    if args.config:
        path = Path(args.config).expanduser().resolve()
        with path.open("r", encoding="utf-8") as handle:
            config.update(_flatten(json.load(handle)))

    for key in ("host", "port", "il_checkpoint", "output", "total_steps",
                "buffer_capacity", "width", "height", "obs_width", "obs_height",
                "batch_size", "device", "action_repeat", "max_episode_steps", "town", "seed"):
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    if args.warm_start is not None:
        config["warm_start"] = args.warm_start
    if args.spectator_follow is not None:
        config["spectator_follow"] = args.spectator_follow
    if args.realtime is not None:
        config["realtime"] = args.realtime

    # `--town A` -> "A" (chuoi, cho evaluate.py), `--town A B` -> ["A","B"] (xoay vong).
    if isinstance(config.get("town"), (list, tuple)) and len(config["town"]) == 1:
        config["town"] = config["town"][0]

    config["_resume"] = args.resume
    config["_episodes"] = args.episodes
    config["_deterministic"] = args.deterministic
    config["_eval_csv_out"] = args.eval_csv_out
    return config
