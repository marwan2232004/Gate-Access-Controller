import React from "react";
import logo from "../assets/eagle.png";

const NavBar = () => {
  return (
    <nav className="bg-gray-800 p-4">
      <div className="container mx-auto flex items-center">
        <div className="flex items-center">
          <img src={logo} alt="Logo" className="h-8 w-8 mr-2" />
          <span className="text-white text-xl font-bold">
            Eagle Vision: Gate Access Controller
          </span>
        </div>
      </div>
    </nav>
  );
};

export default NavBar;
