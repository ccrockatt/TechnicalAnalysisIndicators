from unittest import TestCase

import numpy as np

from technical_analysis_automation.swing_chart_candle_plotter import is_data_point_a_local_swing


class TestSwingChartCandlePlotter(TestCase):
    def test__is_data_point_a_local_swing__should_not_have_enough_data_points(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                             current_index=4,
                                             time_radius=2,
                                             is_top=True)
        print(f"result is {result}")
        assert result is False

    def test__is_data_point_a_local_swing__should_be_a_local_swing_top(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                             current_index=5,
                                             time_radius=1,
                                             is_top=True)
        assert result is True

    def test__is_data_point_a_local_swing__should_be_a_local_2d_wide_swing_top(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                             current_index=5,
                                             time_radius=2,
                                             is_top=True)
        assert result is True

    def test__is_data_point_a_local_swing__should_not_be_a_local_swing_top(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                             current_index=6,
                                             time_radius=2,
                                             is_top=True)
        assert result is False

    def test__is_data_point_a_local_swing__should_not_be_a_local_swing_bottom(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 5, 4, 3, 2, 1]),
                                             current_index=5,
                                             time_radius=1,
                                             is_top=False)
        assert result is False

    def test__is_data_point_a_local_swing__should_be_a_local_swing_bottom(self) -> None:
        result = is_data_point_a_local_swing(data_set=np.array([1, 2, 3, 4, 1, 4, 3, 2, 1]),
                                             current_index=5,
                                             time_radius=1,
                                             is_top=False)
        assert result is False
