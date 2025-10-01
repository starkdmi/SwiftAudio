// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "AudioUtils",
    platforms: [
        .iOS(.v16),
        .macOS("13.3")
    ],
    products: [
        .library(
            name: "AudioUtils",
            targets: ["AudioUtils"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.18.0")
    ],
    targets: [
        .target(
            name: "AudioUtils",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXOptimizers", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift")
            ],
            path: "Sources/AudioUtils"
        )
    ]
)