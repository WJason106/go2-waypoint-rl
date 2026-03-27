# terrain_cfg.py
import isaaclab.terrains as terrain_gen

# ============================================================
# Parkour Terrain Generator Configuration
# ============================================================
# 说明：
# - 该配置严格保持你当前风格：TerrainGeneratorCfg + sub_terrains
# - 目标：减少虚空、提升课程学习可解释性
# - 难度递增依赖 curriculum=True + difficulty_range + 各子地形参数区间
# ============================================================

PARKOUR_TERRAIN_CFG = terrain_gen.TerrainGeneratorCfg(
    # 每个子地形块大小（米）
    size=(8.0, 8.0),
    # 整个地形图外围边界宽度（米）
    border_width=20.0,

    # 地形课程网格：行通常对应难度层级，列对应地形采样
    num_rows=10,
    num_cols=20,

    # HF 相关参数（mesh地形也常保留以兼容pipeline）
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,

    # 课程难度范围：从 0 到 1
    difficulty_range=(0.0, 1.0),

    # 调参阶段建议关闭缓存，避免修改后不生效
    use_cache=False,

    # 子地形池（按 proportion 随机选类型；同一类型内按难度变化参数）
    sub_terrains={
        # ----------------------------------------------------
        # 0) 平地（给 early-stage 稳定探索）
        # ----------------------------------------------------
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10
        ),

        # ----------------------------------------------------
        # 1) 轻度粗糙（低幅随机起伏）
        # ----------------------------------------------------
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10,
            noise_range=(0.005, 0.040),  # 难度增大时可采样到更大起伏
            noise_step=0.005,
            border_width=0.25,
        ),

        # ----------------------------------------------------
        # 2) Ramp 上坡 / 下坡
        # ----------------------------------------------------
        "ramp_up": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.05, 0.42),  # 易->难：坡度增大
            platform_width=2.0,
            border_width=0.25,
        ),
        "ramp_down": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.10,
            slope_range=(0.05, 0.42),
            platform_width=2.0,
            border_width=0.25,
        ),

        # ----------------------------------------------------
        # 3) Hurdle近似：离散箱体障碍
        # ----------------------------------------------------
        # 注意：grid_width 不宜过大，否则可能触发 border width 报错
        "hurdle_approx": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.20,
            grid_width=0.23,                 # 关键：从0.4/0.45降到0.20更稳
            grid_height_range=(0.04, 0.35), # 难度增大时可更高
            platform_width=1.4,              # 关键：减小平台宽避免内部border为0
        ),

        # ----------------------------------------------------
        # 4) Platform 上台 / 下台
        # ----------------------------------------------------
        "platform_up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.30),  # 难度增大台阶更高
            step_width=0.30,
            platform_width=2.6,
            border_width=0.8,
            holes=False,                     # 明确不挖洞，避免“虚空”
        ),
        "platform_down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.15,
            step_height_range=(0.04, 0.30),
            step_width=0.30,
            platform_width=2.6,
            border_width=0.8,
            holes=False,
        ),

        # ----------------------------------------------------
        # 5) Gap近似（不使用holes，避免大块虚空）
        # ----------------------------------------------------
        # 这里用“反向台阶+较大步宽”近似产生跨越需求，不做真实挖空
        "gap_approx_no_hole": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.10,
            step_height_range=(0.05, 0.22),
            step_width=0.45,
            platform_width=2.0,
            border_width=0.8,
            holes=False,  # 关键：去掉虚空
        ),
    },
)