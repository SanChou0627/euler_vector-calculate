import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

plt.rcParams['font.sans-serif']=['Microsoft YaHei']

# 设置文件路径
input_file = "data.txt"  # 原始数据文件

# 地球半径 (mm)
R = 6371000.0 * 1000  # 转换为毫米

#选取经纬度（从子午线开始，向东增加，最大360度）
lon_min = 20
lon_max = 40
lat_min = 20
lat_max = 40

#读取数据
def select_data(input_file):

    lons, lats, ve, vn = [], [], [], []

    with open(input_file, 'r') as f:
        for line in f:
            # 分割每行数据
            columns = line.strip().split()

            # 确保至少有4列数据
            if len(columns) >= 4:
                try:
                    lon = float(columns[0])
                    lat = float(columns[1])
                    v_east = float(columns[2])
                    v_north = float(columns[3])

                    lons.append(lon)
                    lats.append(lat)
                    ve.append(v_east)
                    vn.append(v_north)
                except ValueError:
                    continue

    # 转换为NumPy数组
    lons = np.array(lons)
    lats = np.array(lats)
    ve = np.array(ve)
    vn = np.array(vn)

    return lons, lats, ve, vn


def filter_data(lons, lats, ve, vn, lon_min, lon_max, lat_min, lat_max):

    # 筛选符合条件的点
    mask = (lon_min <= lons) & (lons <= lon_max) & (lat_min <= lats) & (lats <= lat_max)

    filtered_lons = lons[mask]
    filtered_lats = lats[mask]
    filtered_ve = ve[mask]
    filtered_vn = vn[mask]

    print(f"原始数据点: {len(lons)}, 筛选后数据点: {len(filtered_lons)}")
    return filtered_lons, filtered_lats, filtered_ve, filtered_vn


def calculate_euler_vector(lons, lats, ve_obs, vn_obs):

    # 将经纬度转换为弧度
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)

    # 构建设计矩阵 A
    A = np.zeros((2 * len(lons), 3))

    for i in range(len(lons)):
        # 东向速度方程系数
        A[2 * i, 0] = -R * np.sin(lats_rad[i]) * np.cos(lons_rad[i])
        A[2 * i, 1] = -R * np.sin(lats_rad[i]) * np.sin(lons_rad[i])
        A[2 * i, 2] = R * np.cos(lats_rad[i])

        # 北向速度方程系数
        A[2 * i + 1, 0] = -R * np.sin(lons_rad[i])
        A[2 * i + 1, 1] = R * np.cos(lons_rad[i])
        A[2 * i + 1, 2] = 0

    # 构建观测向量 L
    L = np.zeros(2 * len(lons))
    L[0::2] = ve_obs  # 东向速度观测值
    L[1::2] = vn_obs  # 北向速度观测值

    # 最小二乘求解: (A^T A)^-1 A^T L
    ATA = np.dot(A.T, A)
    ATL = np.dot(A.T, L)
    euler_vector = np.linalg.solve(ATA, ATL)

    # 计算旋转极位置
    wx, wy, wz = euler_vector
    rotation_rate = np.sqrt(wx ** 2 + wy ** 2 + wz ** 2)  # 弧度/年

    # 计算旋转极纬度 (度)
    pole_lat = np.degrees(np.arcsin(wz / rotation_rate))

    # 计算旋转极经度 (度)
    pole_lon = np.degrees(np.arctan2(wy, wx))
    if pole_lon < 0:
        pole_lon += 360

    # 旋转速率转换为度/百万年
    rotation_rate_deg_myr = rotation_rate * (180 / np.pi) * 1e6

    return euler_vector, (pole_lon, pole_lat), rotation_rate_deg_myr


def calculate_theoretical_velocity(euler_vector, lons, lats):

    # 将经纬度转换为弧度
    lons_rad = np.radians(lons)
    lats_rad = np.radians(lats)

    wx, wy, wz = euler_vector

    # 初始化理论速度数组
    ve_theo = np.zeros(len(lons))
    vn_theo = np.zeros(len(lons))

    for i in range(len(lons)):
        # 计算理论东向速度
        ve_theo[i] = -wx * np.sin(lats_rad[i]) * np.cos(lons_rad[i]) * R
        ve_theo[i] += -wy * np.sin(lats_rad[i]) * np.sin(lons_rad[i]) * R
        ve_theo[i] += wz * np.cos(lats_rad[i]) * R

        # 计算理论北向速度
        vn_theo[i] = -wx * np.sin(lons_rad[i]) * R
        vn_theo[i] += wy * np.cos(lons_rad[i]) * R

    return ve_theo, vn_theo


def plot_results(lons, lats, rotation_pole, ve_obs, vn_obs, ve_theo, vn_theo):

    # 创建三个子图
    fig = plt.figure(figsize=(18, 12))

    # 子图1: 站点位置和旋转极
    ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.Robinson())
    ax1.set_global()
    ax1.add_feature(cfeature.COASTLINE)
    ax1.add_feature(cfeature.LAND, facecolor='lightgray')
    ax1.add_feature(cfeature.OCEAN, facecolor='white')
    ax1.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # 绘制站点位置
    ax1.scatter(lons, lats, color='blue', s=50, transform=ccrs.PlateCarree(),
                label='站点位置', zorder=5)

    # 绘制旋转极
    pole_lon, pole_lat = rotation_pole
    ax1.scatter(pole_lon, pole_lat, color='red', s=100, transform=ccrs.PlateCarree(),
                marker='*', label='旋转极', zorder=6)

    ax1.set_title('站点位置与欧拉旋转极')
    ax1.legend(loc='lower left')

    # 子图2: 观测速度与理论速度比较
    ax2 = fig.add_subplot(2, 2, 2, projection=ccrs.PlateCarree())
    ax2.add_feature(cfeature.COASTLINE)
    ax2.add_feature(cfeature.BORDERS, linestyle=':')
    ax2.add_feature(cfeature.LAND, facecolor='lightgray')
    ax2.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)

    # 设置合适的视图范围
    margin = 5  # 边界扩展度数
    ax2.set_extent([min(lons) - margin, max(lons) + margin,
                    min(lats) - margin, max(lats) + margin],
                   crs=ccrs.PlateCarree())

    # 绘制观测速度
    q1 = ax2.quiver(lons, lats, ve_obs, vn_obs, color='blue', scale=150,
                    transform=ccrs.PlateCarree(), label='观测速度')

    # 绘制理论速度
    q2 = ax2.quiver(lons, lats, ve_theo, vn_theo, color='red', scale=150,
                    transform=ccrs.PlateCarree(), label='理论速度')

    ax2.quiverkey(q1, X=0.85, Y=1.05, U=10, label='10 mm/yr', labelpos='E')
    ax2.set_title('观测速度 vs 理论速度')
    ax2.legend(loc='lower left')

    # 子图3: 速度残差
    ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
    ax3.add_feature(cfeature.COASTLINE)
    ax3.add_feature(cfeature.BORDERS, linestyle=':')
    ax3.add_feature(cfeature.LAND, facecolor='lightgray')
    ax3.gridlines(draw_labels=True, dms=True, x_inline=False, y_inline=False)
    ax3.set_extent([min(lons) - margin, max(lons) + margin,
                    min(lats) - margin, max(lats) + margin],
                   crs=ccrs.PlateCarree())

    # 计算残差大小
    ve_res = ve_obs - ve_theo
    vn_res = vn_obs - vn_theo
    magnitude_res = np.sqrt(ve_res ** 2 + vn_res ** 2)

    # 绘制残差
    sc = ax3.scatter(lons, lats, c=magnitude_res, cmap='viridis', s=50,
                     transform=ccrs.PlateCarree())

    # 添加残差向量箭头
    ax3.quiver(lons, lats, ve_res, vn_res, color='black', scale=300,
               transform=ccrs.PlateCarree())

    cbar = plt.colorbar(sc, ax=ax3, orientation='vertical', pad=0.05)
    cbar.set_label('速度残差大小 (mm/yr)')
    ax3.set_title('速度残差分布')

    # 子图4: 残差直方图
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.hist(magnitude_res, bins=20, color='skyblue', edgecolor='black')
    ax4.set_xlabel('速度残差大小 (mm/yr)')
    ax4.set_ylabel('频数')
    ax4.set_title('速度残差分布直方图')
    ax4.grid(True, linestyle='--', alpha=0.7)

    # 添加残差统计
    mean_res = np.mean(magnitude_res)
    std_res = np.std(magnitude_res)
    ax4.axvline(mean_res, color='red', linestyle='dashed', linewidth=1)
    ax4.text(mean_res + 0.1, ax4.get_ylim()[1] * 0.9,
             f'均值: {mean_res:.2f} mm/yr', color='red')
    ax4.text(mean_res + 0.1, ax4.get_ylim()[1] * 0.8,
             f'标准差: {std_res:.2f} mm/yr', color='red')

    plt.tight_layout()
    plt.savefig('euler_vector_results.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():


    # 1. 调用select函数选择前四列数据
    print("步骤1: 选择前四列数据")
    lons, lats, ve, vn = select_data(input_file)

    # 2. 设置筛选范围并调用filter函数
    print("\n步骤2: 根据经纬度范围筛选数据")
    # 示例范围：北京市大致范围


    filtered_lons, filtered_lats, filtered_ve, filtered_vn = filter_data(
        lons, lats, ve, vn,
        lon_min, lon_max, lat_min, lat_max,
    )

    # 检查是否有足够的数据点
    if len(filtered_lons) < 3:
        print("错误：需要至少3个点才能计算欧拉矢量")
        return

    print(f"\n步骤3: 使用筛选后的 {len(filtered_lons)} 个点计算欧拉矢量")

    # 3. 计算欧拉矢量
    euler_vector, rotation_pole, rotation_rate = calculate_euler_vector(
        filtered_lons, filtered_lats, filtered_ve, filtered_vn
    )

    # 打印欧拉矢量结果
    print("\n欧拉矢量计算结果:")
    print(f"欧拉矢量 (ωx, ωy, ωz): {euler_vector[0]:.6e}, {euler_vector[1]:.6e}, {euler_vector[2]:.6e} rad/yr")
    print(f"旋转极位置: 经度 = {rotation_pole[0]:.4f}°, 纬度 = {rotation_pole[1]:.4f}°")
    print(f"旋转速率: {rotation_rate:.4f} °/Myr")

    # 4. 计算理论速度
    ve_theo, vn_theo = calculate_theoretical_velocity(euler_vector, filtered_lons, filtered_lats)

    # 5. 计算残差
    ve_res = filtered_ve - ve_theo
    vn_res = filtered_vn - vn_theo
    magnitude_res = np.sqrt(ve_res ** 2 + vn_res ** 2)

    # 计算残差统计
    mean_ve_res = np.mean(ve_res)
    mean_vn_res = np.mean(vn_res)
    rms_res = np.sqrt(np.mean(ve_res ** 2 + vn_res ** 2))
    max_res = np.max(magnitude_res)

    print("\n残差统计:")
    print(f"平均东向速度残差: {mean_ve_res:.4f} mm/yr")
    print(f"平均北向速度残差: {mean_vn_res:.4f} mm/yr")
    print(f"速度残差RMS: {rms_res:.4f} mm/yr")
    print(f"最大速度残差: {max_res:.4f} mm/yr")

    # 6. 保存结果到文件
    result_file = "euler_vector_results.txt"
    with open(result_file, 'w') as f:
        f.write("欧拉矢量计算结果:\n")
        f.write(f"ωx: {euler_vector[0]:.6e} rad/yr\n")
        f.write(f"ωy: {euler_vector[1]:.6e} rad/yr\n")
        f.write(f"ωz: {euler_vector[2]:.6e} rad/yr\n")
        f.write(f"旋转极经度: {rotation_pole[0]:.6f}°\n")
        f.write(f"旋转极纬度: {rotation_pole[1]:.6f}°\n")
        f.write(f"旋转速率: {rotation_rate:.6f} °/Myr\n\n")

        f.write("站点速度比较:\n")
        f.write(
            "经度(deg)  纬度(deg)  东向观测(mm/yr)  北向观测(mm/yr)  东向理论(mm/yr)  北向理论(mm/yr)  东向残差(mm/yr)  北向残差(mm/yr)  残差大小(mm/yr)\n")

        for i in range(len(filtered_lons)):
            f.write(f"{filtered_lons[i]:.6f}  {filtered_lats[i]:.6f}  ")
            f.write(f"{filtered_ve[i]:.4f}  {filtered_vn[i]:.4f}  ")
            f.write(f"{ve_theo[i]:.4f}  {vn_theo[i]:.4f}  ")
            f.write(f"{ve_res[i]:.4f}  {vn_res[i]:.4f}  ")
            f.write(f"{magnitude_res[i]:.4f}\n")

    print(f"\n结果已保存到 '{result_file}'")

    # 7. 可视化结果
    print("\n生成可视化结果...")
    plot_results(filtered_lons, filtered_lats, rotation_pole, filtered_ve, filtered_vn, ve_theo, vn_theo)
    print("可视化结果已保存为 'euler_vector_results.png'")


if __name__ == "__main__":
    # 执行主函数
    main()