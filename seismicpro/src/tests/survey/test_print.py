"""Test Survey printing routines"""


class TestReindex:
    def test_str(self, survey):
        assert str(survey)

    def test_str_prints_stats(self, survey_no_stats):
        no_stats_str = str(survey_no_stats)
        stats_str = str(survey_no_stats.collect_stats())
        assert len(stats_str) > len(no_stats_str)
        assert stats_str.startswith(no_stats_str)

    def test_info_matches_str(self, survey, capsys):
        survey.info()
        stdout = capsys.readouterr().out
        assert str(survey) + "\n" == stdout
