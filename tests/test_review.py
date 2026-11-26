import re, os, unittest
from datetime import date, timedelta
src = open(os.path.join(os.path.dirname(__file__), '..', 'app.py')).read()
ns = {"re": re, "date": date, "timedelta": timedelta}
exec(src[src.index("def parse_when"):src.index("def run_tool")].split("\n\n\ndef run_tool")[0], ns)
exec(src[src.index("def validate_booking"):src.index("def run_tool")], ns)
tm = str(date.today() + timedelta(days=1))
eb, vb, sc = ns["extract_booking"], ns["validate_booking"], ns["safe_calc"]
class T(unittest.TestCase):
    def test_exact_user_sentence(self):
        r = eb("Please book me a reservation for five people in Mainland China, South City Mall for tomorrow at 12:00 PM for lunch.")
        self.assertEqual(r, dict(restaurant="Mainland China, South City Mall", party_size=5, date=tm, time="12:00"))
    def test_tgi_fridays_not_a_date(self):
        r = eb("Book a table for 4 at TGI Fridays at 7 PM")
        self.assertEqual(r["restaurant"], "TGI Fridays"); self.assertIsNone(r["date"])
    def test_real_friday_is_a_date(self):
        self.assertIsNotNone(eb("Book a table for 4 at Saffron on Friday at 7 PM")["date"])
    def test_time_before_venue(self):
        r = eb("Book a table for 2 at 7 PM at Saffron tomorrow")
        self.assertEqual((r["restaurant"], r["time"], r["date"], r["party_size"]), ("Saffron", "19:00", tm, 2))
    def test_explicit_iso_date(self):
        r = eb("Book a table for 3 at Saffron on 2026-12-01 at 8 PM")
        self.assertEqual((r["date"], r["restaurant"], r["time"]), ("2026-12-01", "Saffron", "20:00"))
    def test_missing_fields_stay_none(self):
        r = eb("Book a table at Haldiram's"); self.assertEqual((r["party_size"], r["date"], r["time"]), (None, None, None))
    def test_validation(self):
        ok = dict(party_size=2, date="2026-12-01", time="19:00")
        self.assertIsNone(vb(ok))
        for bad in (dict(ok, party_size=-3), dict(ok, party_size=0), dict(ok, party_size="x"), dict(ok, date="2026-13-45"), dict(ok, date="tomorrow"), dict(ok, time="99:99"), dict(ok, time="7pm")):
            self.assertIsNotNone(vb(bad), bad)
    def test_calc(self):
        self.assertEqual(sc("2+3*4"), 14); self.assertEqual(sc("(10-4)/2"), 3)
        for bad in ("9**9**9", "10**100", "1" * 200, "__import__('os')", "2**1000000", "1e999*1e999"):
            with self.assertRaises(Exception): sc(bad)
if __name__ == "__main__":
    unittest.main(verbosity=1)
