import unittest

from image_generation.image_generator import ImageGenerator


class ImageGeneratorTest(unittest.TestCase):
    def test_altcoin_divergence_excludes_target_close_alias(self):
        generator = ImageGenerator(window_size=30, target_symbol="BTCUSDT")
        cols = generator._get_alt_cols(
            ["Close", "BTCUSDT_Close", "ADAUSDT_Close", "ETHUSDT_Close", "Volume"],
            btc_close_col="Close",
            alt_suffix="_Close",
        )

        self.assertEqual(cols, ["ADAUSDT_Close", "ETHUSDT_Close"])


if __name__ == "__main__":
    unittest.main()
