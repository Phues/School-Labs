import pytest
#import module.py from C:\Users\Benya\SEDS_Lab4\tp_seds4\src\models\module.py
import sys
sys.path.append("C:\\Users\\Benya\\SEDS_Lab4\\tp_seds4\\src\\models")
from module import serve_beer

def test_serve_beer_legal():
  adult = 25
  assert serve_beer(adult) == "Have beer"

def test_serve_beer_illegal():
  child = 10
  assert serve_beer(child) == "No beer"