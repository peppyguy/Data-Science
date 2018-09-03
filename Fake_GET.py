#!flask/bin/python
"""

1st Exercise
Fake Resolution

GET /api/issue/{issue-key}/resolve-fake

{
        'issue' : 'AVRO-9999',
        'predicted_resolution_date' : '1970-01-01T00:00:00.000+0000'
    }

"""

from flask import Flask, jsonify
from flask import abort
from flask import make_response

app = Flask(__name__)



@app.route('/api/issue/<name>/resolve-fake', methods=['GET'])

def get_task(name):
    if len(name) == 0:
        abort(404)
    task={'issue':name, 'predicted_resolution_date': '1984-01-01T00:00:00.000+0000'}
    return jsonify(task)


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

if __name__ == '__main__':
    app.run(debug=True)
