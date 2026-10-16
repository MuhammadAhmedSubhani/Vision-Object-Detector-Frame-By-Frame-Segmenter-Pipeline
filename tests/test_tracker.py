import unittest
from src.tracker import CentroidTracker, box_iou

class TestTracker(unittest.TestCase):
    def test_iou_identical(self):
        self.assertAlmostEqual(box_iou((0,0,10,10),(0,0,10,10)),1.0)
    def test_nearby_object_keeps_id(self):
        t=CentroidTracker()
        a=t.update([{"bbox":(10,10,50,50),"class_name":"car"}])
        b=t.update([{"bbox":(12,11,52,51),"class_name":"car"}])
        self.assertEqual(a[0]["track_id"],b[0]["track_id"])
    def test_different_class_gets_new_id(self):
        t=CentroidTracker()
        a=t.update([{"bbox":(10,10,50,50),"class_name":"car"}])
        b=t.update([{"bbox":(10,10,50,50),"class_name":"bus"}])
        self.assertNotEqual(a[0]["track_id"],b[0]["track_id"])
if __name__=="__main__": unittest.main()
