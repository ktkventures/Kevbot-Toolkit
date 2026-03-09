import setuptools

setuptools.setup(
    name="streamlit-lightweight-charts",
    version="0.8.0.dev0",
    author="RoR Trader (forked from freyastreamlit)",
    description="Vendored fork of streamlit-lightweight-charts with LWC v4.2 + primitives",
    packages=["streamlit_lightweight_charts"],
    package_data={
        "streamlit_lightweight_charts": [
            "frontend/build/*",
            "frontend/build/static/js/*",
        ],
    },
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "streamlit >= 0.62",
    ],
)
