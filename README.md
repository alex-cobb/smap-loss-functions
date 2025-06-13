# smap-loss-functions
SMAP loss functions after Koster et al (2017), doi:10.1175/jhm-d-16-0285.1

## Installation

You can install `smap_loss_functions` using pip:

```bash
pip install smap_loss_functions
```

For development, clone the repository and install in editable mode:

```bash
git clone [https://github.com/alex-cobb/smap-loss-functions.git](https://github.com/alex-cobb/smap-loss-functions.git)
cd smap_loss_functions
pip install -e .
```

## Usage
Choosing an EASE grid for a geographic extent:
```python
smap-loss-functions choose-ease-grid sumatra_bbox.json smap_grid.json \
  --grid-name EASE2_G9k
  --cols smap_cols.tif --rows smap_rows.tif
```

## Running Tests

To run the tests, navigate to the root directory of the project and execute pytest:

```bash
pytest
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for
details.
