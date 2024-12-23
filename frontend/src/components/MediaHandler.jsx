import { useState } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faUpload } from "@fortawesome/free-solid-svg-icons";
import DragAndDrop from "./DragAndDrop";
import MediaUploader from "./MediaUploader";
const MediaHandler = ({ setLoading }) => {
  const [image, setImage] = useState(null);
  return (
    <>
      {!image && <DragAndDrop setImage={setImage} />}
      {image && (
        <MediaUploader
          image={image}
          setImage={setImage}
          setLoading={setLoading}
        />
      )}
    </>
  );
};

export default MediaHandler;
