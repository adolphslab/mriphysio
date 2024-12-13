import unittest
from bidsphysio import bids_stub

class TestBidsPhysio(unittest.TestCase):

    def test_bids_stub_single_extension(self):
        pname = "/path/to/file/sub-01_task-rest_bold.nii"
        expected_stub = "sub-01_task-rest"
        self.assertEqual(bids_stub(pname), expected_stub)

    def test_bids_stub_double_extension(self):
        pname = "/path/to/file/sub-01_task-rest_bold.nii.gz"
        expected_stub = "sub-01_task-rest"
        self.assertEqual(bids_stub(pname), expected_stub)

    def test_bids_stub_no_extension(self):
        pname = "/path/to/file/sub-01_task-rest_bold"
        expected_stub = "sub-01_task-rest"
        self.assertEqual(bids_stub(pname), expected_stub)

    def test_bids_stub_no_underscore(self):
        pname = "/path/to/file/sub01taskrest.nii.gz"
        expected_stub = "sub01taskrest"
        self.assertEqual(bids_stub(pname), expected_stub)

if __name__ == '__main__':
    unittest.main()