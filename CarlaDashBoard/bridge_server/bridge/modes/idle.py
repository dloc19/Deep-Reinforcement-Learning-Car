"""IDLE — khong che do lai nao dang chay, xe dung yen.

VI SAO PHAI CO MOT MODE RIENG chu khong dung thang `ModeRuntime` mac dinh:

1. `ModeRuntime.name` la "BASE", nen telemetry (va o giao dien la dong "Che do") hien
   "BASE" — mot cai ten khong co trong giao thuc, khong co trong danh sach mode cua man
   hinh Settings, va khong noi len dieu gi voi nguoi xem demo.

2. Quan trong hon: `tick()` cua no tra ve None, tuc "khong ap lenh nao". Nhung CARLA GIU
   NGUYEN lenh dieu khien cuoi cung da ap len xe cho toi khi co lenh khac — bo lai lenh do
   khong lam xe dung, ma lam xe chay MAI voi ga cu. Do duoc trong bo test end-to-end: chuyen
   tu ROUTE_DRL_AUTOPILOT sang IDLE, xe van chay tiep va con tang len 52 km/h voi
   throttle=0.65, khong ai lai, cho toi khi dam vao vat gi do.

Nen IDLE phanh va giu phanh. Do cung la hanh vi ma AstarAutopilotMode/RouteLearnedAutopilot
dung khi "da toi dich" (brake=1.0), giu cho y nghia "khong lai nua" nhat quan o moi cho.
"""

import carla

from .base import ModeRuntime


class IdleMode(ModeRuntime):
    name = "IDLE"

    _STOP = carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)

    def tick(self, snapshot):
        return self._STOP
