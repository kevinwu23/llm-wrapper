# tests/test_module1.py

import unittest
from wrappers import GPTWrapper

class TestModule1(unittest.TestCase):
    def gpt_text(self):
        model = GPTWrapper(model_name='gpt4o')
        
        self.assertEqual(1, 1)

if __name__ == '__main__':
    unittest.main()