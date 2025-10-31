curl -X POST localhost:8000/v2/models/embeddinggemma-300m/infer -d \
'{
  "inputs": [
    {
      "name": "TEXT",
      "shape": [ 1, 2 ],
      "datatype": "BYTES",
      "data": [ "This is the first document", "This is the second document" ]
    }
  ]
}'
