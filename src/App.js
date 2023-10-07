import Navbar from './Navbar';
import Button from './Button';
import './App.css'
import Monitor from './Monitor';


function App() {
  return (
    <div className="App">
      <Navbar />
      <div className='Monitor'>
         <Monitor/>
      </div> 
    </div>   
    
  );
}

export default App;