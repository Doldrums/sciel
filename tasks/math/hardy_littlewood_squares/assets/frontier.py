"""The best known construction for this task: the 59-atom half cell of R. Pinho's certificate for C_47 (rpinho/math-
certificates commit 3d4cd55, c47/round1/certificates/half59.json, teorth/optimizationproblems PR 321, 2026; candidate,
unrefereed), further hole-filling on T. S. Tomar's 50-atom measure (zenodo 23716077, C_47 >= 1.7027237335), which
refines Y. Lin's weighted lattice. 50 inherited atoms and 9 inserted ones, rational coordinates and masses, period
Px = 164377561597/50000000000, Py = 2744174966813/1000000000000. Tomar's check_exact2.py and assets/hl.py both give
the ratio 215956002657486428307163448091951082893881/126800499112387565339223240212480000000000 = 1.70311634551278...
from 33356 rectangles. The data is the measure's JSON, zlib-compressed and base64-encoded."""

import base64
import json
import zlib

DATA = (
    "eNqdWEuuIzcMvIvXA5h/inOK2T/MIgcIskgWCYLcPUXn92yp1e4YzwZey12qLlJk0b/fvv16+3rjMM30YK+8O/37un25ffsN65JmnFYRg/XORE9f+eGX"
    "n378+fb14+PW//Wbb9+/fDyj8vM9/RYi0xCR4WGjPm/cX2oIL8IKudTLem9Cw0XdK4bzRCpIzILM+ZXxG9BDygK4VoOnbV0GFkzS9S4+4QpTeCoLiM1a"
    "7Uk3tuBOKPM/sHesgxlXtO/NpRwsJlaqUTIFa09aKQfUHopor2K4Qz6hHDqUOIWXlAXRJcfmEhcpZ4CQlGfd+bPMp7AnfItCVVLWGcdKRjRCxS9nhpJD"
    "itK0F43fQD4TeRjuxOqSdBVnshTuv3oCAS0RBXi9DL3jjOiJ6uNoP4fv7/ghyckygc53uZgWeIlW5lXgE40T8RM86zqRIb6RWyDR8xrj1pgHmw/Oy9Bn"
    "nJNR8OsI2CNR1wf+8qlUvcO5OBD8kOvQZ5W5uLBx8Kp6MsqUs1BK1tXz13oIqWv4weHegp/VjS5FFHmgB4KnDf/8WO9wphJFIIkvAu/4IpWzEhlXeFBf"
    "ha8SS8H1KeHeIEuGVpyqfpQVa9i9tOTOaD3pawVchT1ZXxrBgy3IsPJoPq9nUkwRsqjHml9AdRuJwoU67lNqW2XguojbQbjgaHAmIqioFvl7TLjKBjFu"
    "fl44B93xFazhCg6E87LvI9IN6p5XxAXXLgDEba8uoZ6Ii/JgJXFQg1FuKrzGGMlzPTumCxUcNhKNzpanYYd7oi5UkH7SJS5yDPa53PL+vrboX2nC1i4i"
    "JuNzCLnjydUmGt4DfcIW0YLlTOu99P6+qAiSE6ILc7BqxBvUE0mj7T7qeK6HDXgsbeCFb9gnbKRavCydo+7IZsYgwicU8E+x+iupEMG2ez1aXUnWPgU5"
    "4AfE1kV2B7xjCw0EVgEGdKlB23hlxQAg1wpXEx4gjAHxyDptkLd5667lyDP4Xn0qJA+rV20vawCY+SpjDBCwinrgevfQJyLD7MHO1bIyGoZzTLxog67X"
    "RVbDIHDg9bbIZ0UMlxPFYTneC0wEngcduq3eVc4Jd1pH9XwPfdYqMMVhqdbQsIIEH4jGmbM/DdyYeG5E3561QEJYoroemKZBOECe5DUVH/DBrOOY/3hO"
    "8qRHpfAuaut8Q6KNRJumuQBxSwCh/KXJ/DN+wzxicl+rYGjEeCPdcxbYBM/DjNmEdTp6IujuGAyzlh0eA5YbdSXiGVgwYnehwcaTUD0TwqP36tKQMMpb"
    "xwaTzSIluH/d4vTX7tDAEuiM1j9gzG2zYVEy0amz5gM9HHYO6TJIJ4nR4LAfUomWBgr1AbZCB4nS4nQkmjnmHVde1BGwhN8Oe/lB7z+NR/cBHyKxqMjw"
    "zYlpRCHJIgJIcILdybE4Ht//+BOnxvho")


def candidate() -> dict:
    """{"Px": "p/q", "Py": "p/q", "atoms": [["x", "y", "m"], ...]}."""
    return json.loads(zlib.decompress(base64.b64decode("".join(DATA))))
