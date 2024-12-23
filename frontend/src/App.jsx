import { useState } from "react";

import MediaHandler from "./components/MediaHandler";
import NavBar from "./components/NavBar";
import { PulseLoader } from "react-spinners";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
function App() {
  const [loading, setLoading] = useState(false);
  return (
    <div className="h-screen flex flex-col">
      <NavBar />
      <div className="flex-grow text-center bg-slate-600 flex items-center justify-center">
        <MediaHandler setLoading={setLoading} />
        {loading && (
          <div className="absolute inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[60]">
            <PulseLoader color={"#ffffff"} loading={loading} size={30} />
          </div>
        )}
        <ToastContainer />
      </div>
    </div>
  );
}

export default App;
