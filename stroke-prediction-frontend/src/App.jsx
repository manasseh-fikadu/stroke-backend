import "./App.css";
import StrokePredictionForm from "./components/StrokePredictionForm";

function App() {
  return (
    <>
      <h1>Check if you will have a stroke 🧑‍⚕️</h1>
      <div className="card">
        <StrokePredictionForm />
      </div>
    </>
  );
}

export default App;
