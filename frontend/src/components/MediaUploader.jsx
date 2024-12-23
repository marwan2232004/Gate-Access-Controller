import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload } from "@fortawesome/free-solid-svg-icons";
import { toast } from "react-toastify";
const MediaUploader = ({ image, setImage, setLoading }) => {
  const uploadMedia = () => {
    console.log("Uploading media...");
    setLoading(true);
    try {
      fetch("http://localhost:8000/is-allowed", {
        method: "POST",
        body: new FormData().append("image", image),
      })
        .then((res) => res.json())
        .then((data) => {
          console.log(data);
          // if (data.success) {
          //   toast.success("Media uploaded successfully!");
          // } else {
          //   toast.error("Failed to upload media!");
          // }
        });
    } catch (err) {
      console.log(err);
    } finally {
      setLoading(false);
      setImage(null);
    }
  };
  return (
    <div className="w-[400px] h-[400px] max-w-lg mx-auto relative">
      <button
        className="absolute top-[-16px] right-[12px] bg-red-600 text-white rounded-full p-1 hover:bg-red-400 transition-all duration-300 w-8 h-8 flex items-center justify-center z-50"
        onClick={() => setImage(null)}
      >
        X
      </button>
      <div className="relative w-full h-full rounded-lg overflow-hidden group">
        <img
          src={URL.createObjectURL(image)}
          alt="uploaded"
          className="rounded-lg w-full h-full object-fill"
        />
        <div
          className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300 hover:cursor-pointer"
          onClick={uploadMedia}
        >
          <FontAwesomeIcon icon={faUpload} className="text-white text-4xl" />
        </div>
      </div>
    </div>
  );
};

export default MediaUploader;
