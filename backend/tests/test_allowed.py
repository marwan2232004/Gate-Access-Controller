import os
from unittest import TestCase
from io import BytesIO
from fastapi import UploadFile
from app.controller import is_allowed
from app.database import get_db

class TestIsAllowed(TestCase):

    def setUp(self):
        self.db = next(get_db())

    def test_is_allowed(self):
        test_image = "Cars0.png"
        test_image_path = os.path.join(os.path.dirname(__file__), test_image)
        
        if not os.path.exists(test_image_path):
            raise FileNotFoundError(f"Test image {test_image_path} not found.")

        with open(test_image_path, "rb") as image_file:
            image_data = image_file.read()

        image = UploadFile(filename=test_image, file=BytesIO(image_data))

        response = is_allowed(self.db, image)

        json_response = response
        self.assertEqual(json_response["allowed"], False)
        self.assertEqual(json_response["plate"], "ABC1234")
