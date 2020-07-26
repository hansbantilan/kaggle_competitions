##########################################################################################
# Unit tests for the KagglePipeline class methods
##########################################################################################
from KagglePipeline import * # KagglePipeline class
import unittest

class TestKagglePipeline(unittest.TestCase):    

	def setUp(self):
		self.test_object = KagglePipeline('train.csv')
		self.test_targets = [1, 0]

	def test_convert_to_cats(self):
		self.assertEqual(self.test_object.convert_to_cats(self.test_targets), 
		[{'cats': {"real": True, "not": False}}, {'cats': {"real": False, "not": True}}])

if __name__ == "__main__":
    unittest.main(verbosity=2)
