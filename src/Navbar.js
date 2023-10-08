import React from 'react';
import img1 from './images/prepr-logos.jpeg';

const Navbar = () => {
  return (
    <nav className="navbar">
      <div><img src = {img1} alt = "" style={{width: '100px', height: '100px'}}/></div>
      <h1 style = {{textAlign: 'center'}}>Mastering Interviews with AI!</h1>
      <div className="links">
        {/* <a href="/create" style ={{
            color: 'white', backgroundColor: '#f1356d', borderRadius: '8px', textAlign: 'left'
         }}>History</a> */}
      </div>
    </nav>
  );
}

export default Navbar;