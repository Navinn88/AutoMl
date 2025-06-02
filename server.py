from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from urllib.parse import parse_qs, urlparse
import sys
from typing import Dict, Any, Tuple
import numpy as np

from data_utils import prepare_data
from model import LinearRegression

# Global model instance
model = LinearRegression()

class LinearRegressionHandler(BaseHTTPRequestHandler):
    def _send_json_response(self, data: Dict[str, Any], status: int = 200) -> None:
        """Send JSON response with appropriate headers."""
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _send_error_response(self, message: str, status: int = 400) -> None:
        """Send error response in JSON format."""
        self._send_json_response({'error': message}, status)

    def _parse_json_body(self) -> Dict[str, Any]:
        """Parse JSON request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        
        body = self.rfile.read(content_length)
        try:
            return json.loads(body.decode())
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in request body")

    def _parse_csv_body(self) -> str:
        """Parse CSV request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            raise ValueError("Empty request body")
        
        body = self.rfile.read(content_length)
        return body.decode()

    def do_POST(self) -> None:
        """Handle POST requests for training and prediction."""
        try:
            path = urlparse(self.path).path

            if path == '/train':
                self._handle_train()
            elif path == '/predict':
                self._handle_predict()
            else:
                self._send_error_response("Endpoint not found", 404)

        except ValueError as e:
            self._send_error_response(str(e), 400)
        except Exception as e:
            self._send_error_response(f"Internal server error: {str(e)}", 500)

    def _handle_train(self) -> None:
        """Handle training endpoint."""
        try:
            # Parse CSV data
            csv_text = self._parse_csv_body()
            
            # Prepare data
            X, y = prepare_data(csv_text)
            
            # Train model
            global model
            model = LinearRegression()  # Reset model
            model.fit(X, y)
            
            # Get parameters
            params = model.get_params()
            
            # Send response
            self._send_json_response({
                'intercept': float(params['intercept']),
                'coefficients': params['weights'].tolist(),
                'feature_count': len(params['weights'])
            })
            
        except ValueError as e:
            self._send_error_response(f"Training error: {str(e)}", 400)

    def _handle_predict(self) -> None:
        """Handle prediction endpoint."""
        try:
            # Parse JSON body
            data = self._parse_json_body()
            
            if 'features' not in data:
                raise ValueError("Missing 'features' in request body")
            
            features = data['features']
            
            # Validate features
            if not isinstance(features, list):
                raise ValueError("'features' must be a list")
            
            # Convert to numpy array
            X = np.array(features)
            
            # Validate dimensions
            if model.weights is None:
                raise ValueError("Model not trained yet")
            
            expected_features = len(model.weights)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            
            if X.shape[1] != expected_features:
                raise ValueError(
                    f"Expected {expected_features} features, got {X.shape[1]}"
                )
            
            # Make predictions
            predictions = model.predict(X)
            
            # Send response
            self._send_json_response({
                'predictions': predictions.tolist()
            })
            
        except ValueError as e:
            self._send_error_response(f"Prediction error: {str(e)}", 400)
        except np.linalg.LinAlgError as e:
            self._send_error_response(f"Numerical error in prediction: {str(e)}", 400)

def run_server(port: int = 8000) -> None:
    """Run the HTTP server."""
    server_address = ('', port)
    httpd = HTTPServer(server_address, LinearRegressionHandler)
    print(f"Server running on port {port}")
    print("Available endpoints:")
    print("  POST /train  - Train model with CSV data")
    print("  POST /predict - Make predictions with JSON features")
    httpd.serve_forever()

if __name__ == "__main__":
    run_server() 