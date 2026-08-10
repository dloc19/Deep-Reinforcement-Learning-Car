from agents.navigation.global_route_planner import GlobalRoutePlanner
import carla
import random

def draw_blue_route(vehicle, destination=None, life_time=60.0):
    """
    Tính toán và vẽ tuyến đường màu xanh từ xe đến điểm đến.
    - vehicle: Xe ô tô của bạn (ego_car)
    - destination: Vị trí đích (carla.Location). Nếu None, tự chọn ngẫu nhiên.
    - life_time: Thời gian hiển thị tuyến đường trên màn hình (giây).
    """
    if not vehicle or not vehicle.is_alive:
        print("Xe không tồn tại!")
        return
        
    # 1. Khởi tạo bộ tìm đường (Global Route Planner)
    carla_map = world.get_map()
    sampling_resolution = 2.0  # Khoảng cách giữa các điểm màu xanh (mét)
    grp = GlobalRoutePlanner(carla_map, sampling_resolution)
    
    # 2. Lấy vị trí hiện tại của xe
    start_location = vehicle.get_location()
    
    # 3. Xác định điểm đến
    if destination is None:
        spawn_points = carla_map.get_spawn_points()
        destination = random.choice(spawn_points).location
        
    # 4. Tính toán tuyến đường (trả về list các Waypoint)
    route = grp.trace_route(start_location, destination)
    
    # 5. Dùng DebugHelper để vẽ tuyến đường
    for waypoint, road_option in route:
        # Nâng toạ độ Z lên một chút (ví dụ 0.5m) để điểm vẽ không bị chìm dưới mặt đường
        draw_loc = waypoint.transform.location + carla.Location(z=0.5)
        
        # Vẽ mũi tên màu xanh dương (như trong ảnh của bạn)
        world.debug.draw_arrow(
            begin=draw_loc, 
            end=draw_loc + waypoint.transform.get_forward_vector() * 1.5, # Hướng mũi tên
            thickness=0.1, 
            arrow_size=0.15, 
            color=carla.Color(r=0, g=0, b=255), # RGB: Xanh dương
            life_time=life_time
        )
        
    print(f"✓ Đã vẽ tuyến đường màu xanh gồm {len(route)} điểm.")
    return route