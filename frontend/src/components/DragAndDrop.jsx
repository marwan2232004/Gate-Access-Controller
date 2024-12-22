import { useState } from "react";

const FileUpload = ({ setImage }) => {
  const [isDragOver, setIsDragOver] = useState(false);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = () => {
    setIsDragOver(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragOver(false);
    const file = e.dataTransfer.files[0];
    setImage(file);
  };

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    console.log(file);
    setImage(file);
  };

  return (
    <div className="w-1/2 max-w-lg mx-auto">
      <div
        className={`relative border-2 border-dashed rounded-lg p-10 text-center bg-white transition-colors duration-300 ${
          isDragOver ? "bg-blue-500 text-white" : "bg-white"
        } h-80 flex flex-col justify-center`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <input
          type="file"
          id="fileInput"
          accept="image/*"
          className="hidden"
          onChange={handleFileChange}
        />
        <label
          htmlFor="fileInput"
          className="flex flex-col items-center cursor-pointer"
        >
          <span className="text-5xl text-blue-500 mb-8">🌄</span>
          <p className="text-gray-600">Drag & Drop your Image Input</p>
        </label>
      </div>
    </div>
  );
};

export default FileUpload;
