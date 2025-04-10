from flask import Flask, jsonify
import requests

app = Flask(__name__)

BASE_URL = "https://jsonplaceholder.typicode.com/comments"

@app.route('/api/comments/<int:comment_id>', methods=['GET'])
def get_comment(comment_id):
    url = f"{BASE_URL}/{comment_id}" 
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        if response.status_code == 200:
            data = response.json()
            return jsonify({"email": data.get("email", "No email available")})
        else:
            return jsonify({"error": "Comment not found"}), 404

    except requests.exceptions.RequestException as e:
        return jsonify({"error": "Failed to fetch data", "details": str(e)}), 500


@app.route('/')
def home():
    return jsonify({"message": "Flask API is running!"})

@app.route('/favicon.ico')
def favicon():
    return '', 204 

if __name__ == '__main__':
    app.run(debug=True)




#----------------------------------------------------------------------------------











from flask import Flask, jsonify, request
import requests

app = Flask(__name__)

# Dummy user database (in-memory)
users = {
    1: {"name": "Alice", "age": 25},
    2: {"name": "Bob", "age": 30}
}

# Home route
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to my Flask API!"})

# 1️⃣ Get user by ID (GET)
@app.route('/api/user/<int:user_id>', methods=['GET'])
def get_user(user_id):
    user = users.get(user_id)
    if user:
        return jsonify(user)
    return jsonify({"error": "User not found"}), 404

# 2️⃣ Create a new user (POST)
@app.route('/api/user', methods=['POST'])
def create_user():
    data = request.json
    user_id = len(users) + 1
    users[user_id] = {"name": data["name"], "age": data["age"]}
    return jsonify({"message": "User added", "user_id": user_id}), 201

# 3️⃣ Update user details (PUT)
@app.route('/api/user/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404

    data = request.json
    users[user_id].update(data)
    return jsonify({"message": "User updated", "user": users[user_id]})

# 4️⃣ Delete a user (DELETE)
@app.route('/api/user/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    if user_id not in users:
        return jsonify({"error": "User not found"}), 404

    del users[user_id]
    return jsonify({"message": "User deleted"})

# 5️⃣ Fetch random user from external API (GET)
@app.route('/api/random-user', methods=['GET'])
def get_random_user():
    response = requests.get("https://randomuser.me/api/")
    if response.status_code == 200:
        data = response.json()
        user = data["results"][0]
        return jsonify({
            "name": user["name"]["first"] + " " + user["name"]["last"],
            "email": user["email"],
            "location": user["location"]["country"]
        })
    return jsonify({"error": "Failed to fetch data"}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True)
