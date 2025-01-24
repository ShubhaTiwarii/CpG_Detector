{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "G4T6QHHOnfcQ"
   },
   "source": [
    "# Part 1: Build CpG Detector\n",
    "\n",
    "Here we have a simple problem, given a DNA sequence (of N, A, C, G, T), count the number of CpGs in the sequence (consecutive CGs).\n",
    "\n",
    "We have defined a few helper functions / parameters for performing this task.\n",
    "\n",
    "We need you to build a LSTM model and train it to complish this task in PyTorch.\n",
    "\n",
    "A good solution will be a model that can be trained, with high confidence in correctness."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 294,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: torch in c:\\users\\shubh\\anaconda3\\lib\\site-packages (2.5.1)\n",
      "Requirement already satisfied: filelock in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (3.13.1)\n",
      "Requirement already satisfied: typing-extensions>=4.8.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (4.12.2)\n",
      "Requirement already satisfied: networkx in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (3.3)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (3.1.4)\n",
      "Requirement already satisfied: fsspec in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (2024.9.0)\n",
      "Requirement already satisfied: sympy==1.13.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from torch) (1.13.1)\n",
      "Requirement already satisfied: mpmath<1.4,>=1.1.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from sympy==1.13.1->torch) (1.3.0)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jinja2->torch) (2.1.3)\n",
      "Note: you may need to restart the kernel to use updated packages.\n"
     ]
    }
   ],
   "source": [
    "pip install torch"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 295,
   "metadata": {
    "id": "mfS4cLmZD2oB"
   },
   "outputs": [],
   "source": [
    "from typing import Sequence\n",
    "from functools import partial\n",
    "import random\n",
    "import torch\n",
    "import numpy as np\n",
    "import random"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 296,
   "metadata": {
    "id": "_f-brPAvKvTn"
   },
   "outputs": [],
   "source": [
    "# DO NOT CHANGE HERE\n",
    "def set_seed(seed=13):\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)\n",
    "    torch.manual_seed(seed)\n",
    "    if torch.cuda.is_available():\n",
    "        torch.cuda.manual_seed_all(seed)\n",
    "\n",
    "set_seed(13)\n",
    "#print(random.randint(0, 10))  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 297,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Random sequence (integers): [2, 2, 1, 1, 1, 1, 1, 1, 0, 4]\n",
      "Random sequence (integers): [1, 2, 0, 3, 1, 4, 0, 2, 1, 0]\n"
     ]
    }
   ],
   "source": [
    "# Use this for getting x label\n",
    "def rand_sequence(n_seqs: int, seq_len: int=128) -> Sequence[int]:\n",
    "    for i in range(n_seqs):\n",
    "        yield [random.randint(0, 4) for _ in range(seq_len)]\n",
    "\n",
    "\n",
    "# Generates random DNA sequences as lists of integers (0-4). Each integer corresponds to a DNA base.\n",
    "for seq in rand_sequence(2, 10):\n",
    "    print(\"Random sequence (integers):\", seq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 298,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Number of CpGs in 'ACGCGTGACGATATCGGCCG': 5\n"
     ]
    }
   ],
   "source": [
    "# Use this for getting y label\n",
    "def count_cpgs(seq: str) -> int:\n",
    "    cgs = 0\n",
    "    for i in range(0, len(seq) - 1):\n",
    "        dimer = seq[i:i+2]\n",
    "        # note that seq is a string, not a list\n",
    "        if dimer == \"CG\":\n",
    "            cgs += 1\n",
    "    return cgs\n",
    "\n",
    "# Counts the number of consecutive CG pairs in a DNA sequence.\n",
    "print(\"Number of CpGs in 'ACGCGTGACGATATCGGCCG':\", count_cpgs(\"ACGCGTGACGATATCGGCCG\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 299,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "DNA for [0, 1, 2, 3, 4]: NACGT\n",
      "Integers for 'NACGT': [0, 1, 2, 3, 4]\n"
     ]
    }
   ],
   "source": [
    "# Alphabet helpers : Conversion between integer sequences and DNA Sequences\n",
    "alphabet = 'NACGT'\n",
    "dna2int = { a: i for a, i in zip(alphabet, range(5))}\n",
    "int2dna = { i: a for a, i in zip(alphabet, range(5))}\n",
    "\n",
    "intseq_to_dnaseq = partial(map, int2dna.get)\n",
    "dnaseq_to_intseq = partial(map, dna2int.get)\n",
    "\n",
    "print(\"DNA for [0, 1, 2, 3, 4]:\", \"\".join(intseq_to_dnaseq([0, 1, 2, 3, 4])))  # Expected: \"NACGT\"\n",
    "print(\"Integers for 'NACGT':\", list(dnaseq_to_intseq(\"NACGT\")))  # Expected: [0, 1, 2, 3, 4]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 300,
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "executionInfo": {
     "elapsed": 383,
     "status": "ok",
     "timestamp": 1651686469847,
     "user": {
      "displayName": "Ylex",
      "userId": "01820639168093643789"
     },
     "user_tz": 240
    },
    "id": "VK9Qg5GHYxOb",
    "outputId": "0a00bbb6-d9ac-4cf8-ed84-b55b335d7f51"
   },
   "outputs": [],
   "source": [
    "# we prepared two datasets for training and evaluation\n",
    "# training data scale we set to 2048\n",
    "# we test on 512\n",
    "# PREPARING TRAINING AND TEST DATASET\n",
    "def prepare_data(num_samples=100):\n",
    "    \n",
    "    # Step 1: Generates random sequences\n",
    "    X_dna_seqs_train = list(rand_sequence(num_samples))\n",
    "    \n",
    "    # Step 2: Translates these sequences into DNA strings so count_cpgs can calculate the CpG counts\n",
    "    temp = [\"\".join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train] \n",
    "    # intseq_to_dnaseq used to convert ids back to DNA seqs\n",
    "    \n",
    "    # Step 3: Label y_dna_seqs are generated for training\n",
    "    y_dna_seqs = [count_cpgs(seq) for seq in temp] \n",
    "    \n",
    "    return X_dna_seqs_train, y_dna_seqs\n",
    "    \n",
    "train_x, train_y = prepare_data(2048)\n",
    "test_x, test_y = prepare_data(512)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 301,
   "metadata": {},
   "outputs": [],
   "source": [
    "#train_x, train_y = prepare_data(10)\n",
    "#print(\"Sample input sequences (int):\", train_x[:2])\n",
    "#print(\"Sample labels (CpG counts):\", train_y[:2])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 302,
   "metadata": {},
   "outputs": [],
   "source": [
    "# some configurations\n",
    "LSTM_HIDDEN = 64\n",
    "LSTM_LAYER = 3\n",
    "batch_size = 32\n",
    "learning_rate = 0.001\n",
    "epoch_num = 10"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 303,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create data loader\n",
    "from torch.utils.data import Dataset, DataLoader\n",
    "\n",
    "#Wraps X and y for PyTorch so they can be processed in batches\n",
    "class CpGDataset(Dataset):\n",
    "    def __init__(self, X, y):\n",
    "        self.X = torch.tensor(X, dtype=torch.long)  # Inputs DNA sequences\n",
    "        self.y = torch.tensor(y, dtype=torch.float32)  # Labels CpG counts\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.X)\n",
    "\n",
    "    def __getitem__(self, idx):\n",
    "        return self.X[idx], self.y[idx]\n",
    "\n",
    "# Create datasets\n",
    "train_dataset = CpGDataset(train_x, train_y)\n",
    "test_dataset = CpGDataset(test_x, test_y)\n",
    "\n",
    "train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)\n",
    "test_Data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 304,
   "metadata": {
    "id": "q8fgxrM0LnLy"
   },
   "outputs": [],
   "source": [
    "# Model\n",
    "class CpGPredictor(torch.nn.Module):\n",
    "    ''' Simple model that uses a LSTM to count the number of CpGs in a sequence '''\n",
    "    def __init__(self):\n",
    "        super(CpGPredictor, self).__init__()\n",
    "        self.embedding = torch.nn.Embedding(5, 8)  # Embed DNA integers to 8D vectors\n",
    "        # We do need a lstm and a classifier layer here but you are free to implement them in your way\n",
    "        self.lstm = torch.nn.LSTM(input_size=8, hidden_size=LSTM_HIDDEN, num_layers=LSTM_LAYER, batch_first=True)\n",
    "        self.classifier = torch.nn.Linear(LSTM_HIDDEN, 1)  # Outputs a single value (regression)\n",
    "\n",
    "\n",
    "    def forward(self, x):\n",
    "        embedded = self.embedding(x)  # Convert integer sequences to embeddings\n",
    "        _, (hidden, _) = self.lstm(embedded)  # Pass through LSTM\n",
    "        output = self.classifier(hidden[-1])  # Use the last hidden state\n",
    "        return output.squeeze()  # Remove extra dimension\n",
    "        return logits\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 305,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model architecture: CpGPredictor(\n",
      "  (embedding): Embedding(5, 8)\n",
      "  (lstm): LSTM(8, 64, num_layers=3, batch_first=True)\n",
      "  (classifier): Linear(in_features=64, out_features=1, bias=True)\n",
      ")\n"
     ]
    }
   ],
   "source": [
    "model = CpGPredictor()\n",
    "print(\"Model architecture:\", model)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 306,
   "metadata": {},
   "outputs": [],
   "source": [
    "# init model / loss function / optimizer etc.\n",
    "model = CpGPredictor()\n",
    "loss_fn = torch.nn.MSELoss()  # Mean Squared Error\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 307,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/10, Loss: 1852.6523\n",
      "Epoch 2/10, Loss: 1852.6523\n",
      "Epoch 3/10, Loss: 1852.6523\n",
      "Epoch 4/10, Loss: 1852.6523\n",
      "Epoch 5/10, Loss: 1852.6523\n",
      "Epoch 6/10, Loss: 1852.6523\n",
      "Epoch 7/10, Loss: 1852.6523\n",
      "Epoch 8/10, Loss: 1852.6523\n",
      "Epoch 9/10, Loss: 1852.6523\n",
      "Epoch 10/10, Loss: 1852.6523\n"
     ]
    }
   ],
   "source": [
    "# training \n",
    "t_loss = .0\n",
    "model.train()\n",
    "model.zero_grad()\n",
    "for epoch in range(epoch_num):\n",
    "    t_loss = 0.0\n",
    "    for batch in train_data_loader:\n",
    "        x_batch, y_batch = batch\n",
    "        optimizer.zero_grad()\n",
    "        outputs = model(x_batch)\n",
    "        loss = loss_fn(outputs, y_batch)\n",
    "        optimizer.step()\n",
    "        t_loss += loss.item()\n",
    "        loss.backward()\n",
    "\n",
    "    print(f\"Epoch {epoch + 1}/{epoch_num}, Loss: {t_loss:.4f}\")\n",
    "    t_loss = .0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 308,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: pandas in c:\\users\\shubh\\anaconda3\\lib\\site-packages (2.2.3)Note: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Requirement already satisfied: numpy>=1.23.2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas) (2.2.1)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas) (2023.3)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas) (1.17.0)\n"
     ]
    }
   ],
   "source": [
    "pip install pandas\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 309,
   "metadata": {
    "scrolled": True
   },
   "outputs": [],
   "source": [
    "# eval (you can modify the code below)\n",
    "model.eval()\n",
    "\n",
    "res_gs = []\n",
    "res_pred = []\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch in test_Data_loader:\n",
    "        x_batch, y_batch = batch\n",
    "        outputs = model(x_batch)\n",
    "        res_gs.extend(y_batch.tolist())  # Ground truth labels\n",
    "        res_pred.extend(outputs.tolist()) \n",
    "\n",
    "\n",
    "    # TODO complete inference loop"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 310,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error: 4.9281\n"
     ]
    }
   ],
   "source": [
    "# TODO complete evaluation of the model\n",
    "from sklearn.metrics import mean_absolute_error\n",
    "mae = mean_absolute_error(res_gs, res_pred)\n",
    "print(f\"Mean Absolute Error: {mae:.4f}\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {
    "id": "TMrRf_aVDRJm"
   },
   "source": [
    "# Part 2: what if the DNA sequences are not the same length"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 312,
   "metadata": {},
   "outputs": [],
   "source": [
    "# hint we will need following imports\n",
    "from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence\n",
    "from torch.utils.data import DataLoader\n",
    "import torch\n",
    "import random\n",
    "from functools import partial\n",
    "from typing import Sequence"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 313,
   "metadata": {
    "id": "AKvG-MNuXJr9"
   },
   "outputs": [],
   "source": [
    "# DO NOT CHANGE HERE\n",
    "random.seed(13)\n",
    "\n",
    "# Use this for getting x label\n",
    "def rand_sequence_var_len(n_seqs: int, lb: int=16, ub: int=128) -> Sequence[int]:\n",
    "    for i in range(n_seqs):\n",
    "        seq_len = random.randint(lb, ub)\n",
    "        yield [random.randint(1, 5) for _ in range(seq_len)]\n",
    "\n",
    "\n",
    "# Use this for getting y label\n",
    "def count_cpgs(seq: str) -> int:\n",
    "    cgs = 0\n",
    "    for i in range(0, len(seq) - 1):\n",
    "        dimer = seq[i:i+2]\n",
    "        # note that seq is a string, not a list\n",
    "        if dimer == \"CG\":\n",
    "            cgs += 1\n",
    "    return cgs\n",
    "\n",
    "\n",
    "# Alphabet helpers   \n",
    "alphabet = 'NACGT'\n",
    "dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}\n",
    "int2dna = {i: a for a, i in zip(alphabet, range(1, 6))}\n",
    "dna2int.update({\"pad\": 0})\n",
    "int2dna.update({0: \"<pad>\"})\n",
    "\n",
    "intseq_to_dnaseq = partial(map, int2dna.get)\n",
    "dnaseq_to_intseq = partial(map, dna2int.get)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 314,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO complete the task based on the change\n",
    "def prepare_data(num_samples=100, min_len=16, max_len=128):\n",
    "    # TODO prepared the training and test data\n",
    "    # you need to call rand_sequence and count_cpgs here to create the dataset\n",
    "    #step 1\n",
    "    X_dna_seqs_train = list(rand_sequence_var_len(num_samples, min_len, max_len))\n",
    "    \n",
    "    #step 2\n",
    "    temp = [\"\".join(intseq_to_dnaseq(seq)) for seq in X_dna_seqs_train]\n",
    "    \n",
    "    #step3\n",
    "    y_dna_seqs = [count_cpgs(seq) for seq in temp]\n",
    "    \n",
    "    return X_dna_seqs_train, y_dna_seqs\n",
    "    \n",
    "    \n",
    "min_len, max_len = 64, 128\n",
    "train_x, train_y = prepare_data(2048, min_len, max_len)\n",
    "test_x, test_y = prepare_data(512, min_len, max_len)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 315,
   "metadata": {},
   "outputs": [],
   "source": [
    "class MyDataset(torch.utils.data.Dataset):\n",
    "    def __init__(self, lists, labels) -> None:\n",
    "        self.lists = lists\n",
    "        self.labels = labels\n",
    "\n",
    "    def __getitem__(self, index):\n",
    "        return torch.LongTensor(self.lists[index]), self.labels[index]\n",
    "\n",
    "    def __len__(self):\n",
    "        return len(self.lists)\n",
    "\n",
    "\n",
    "    \n",
    "# this will be a collate_fn for dataloader to pad sequence  \n",
    "class PadSequence:\n",
    "    def __call__(self, batch):\n",
    "        # Separate sequences and labels\n",
    "        sequences, labels = zip(*batch)\n",
    "\n",
    "        # Pad sequences to the maximum length in the batch\n",
    "        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)\n",
    "\n",
    "        # Compute original lengths of sequences\n",
    "        lengths = torch.tensor([len(seq) for seq in sequences], dtype=torch.long)\n",
    "\n",
    "        return sequences_padded, lengths, torch.tensor(labels, dtype=torch.float)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 316,
   "metadata": {},
   "outputs": [],
   "source": [
    "# TODO complete the rest"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 317,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create datasets and dataloaders\n",
    "train_dataset = MyDataset(train_x, train_y)\n",
    "test_dataset = MyDataset(test_x, test_y)\n",
    "\n",
    "train_data_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=PadSequence())\n",
    "test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=PadSequence())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 318,
   "metadata": {},
   "outputs": [],
   "source": [
    "# LSTM-based model\n",
    "class CpGPredictor(torch.nn.Module):\n",
    "    def __init__(self, input_size, hidden_size, num_layers, output_size):\n",
    "        super(CpGPredictor, self).__init__()\n",
    "        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)\n",
    "        self.fc = torch.nn.Linear(hidden_size, output_size)\n",
    "\n",
    "    def forward(self, x, lengths):\n",
    "        # Pack padded sequence\n",
    "        packed_input = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)\n",
    "        packed_output, (hidden, _) = self.lstm(packed_input)\n",
    "\n",
    "        # Use the hidden state from the last layer\n",
    "        output = self.fc(hidden[-1])\n",
    "        return output.squeeze()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 319,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Model initialization\n",
    "input_size = 6  # One-hot encoding for DNA bases (N, A, C, G, T, pad)\n",
    "hidden_size = 64\n",
    "num_layers = 2\n",
    "output_size = 1\n",
    "\n",
    "model = CpGPredictor(input_size, hidden_size, num_layers, output_size)\n",
    "loss_fn = torch.nn.MSELoss()\n",
    "optimizer = torch.optim.Adam(model.parameters(), lr=0.001)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 320,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1, Loss: 7.1506461054086685\n",
      "Epoch 2, Loss: 3.6534320898354053\n",
      "Epoch 3, Loss: 3.6483981050550938\n",
      "Epoch 4, Loss: 3.646493498235941\n",
      "Epoch 5, Loss: 3.6563622783869505\n",
      "Epoch 6, Loss: 3.652486763894558\n",
      "Epoch 7, Loss: 3.6735380347818136\n",
      "Epoch 8, Loss: 3.662466386333108\n",
      "Epoch 9, Loss: 3.6443816535174847\n",
      "Epoch 10, Loss: 3.6507404148578644\n"
     ]
    }
   ],
   "source": [
    "# Training loop\n",
    "for epoch in range(10):\n",
    "    model.train()\n",
    "    total_loss = 0.0\n",
    "\n",
    "    for batch_x, batch_lengths, batch_y in train_data_loader:\n",
    "        optimizer.zero_grad()\n",
    "\n",
    "        # One-hot encoding of input sequences\n",
    "        batch_x = torch.nn.functional.one_hot(batch_x, num_classes=6).float()\n",
    "\n",
    "        # Forward pass\n",
    "        outputs = model(batch_x, batch_lengths)\n",
    "\n",
    "        # Compute loss\n",
    "        loss = loss_fn(outputs, batch_y)\n",
    "        loss.backward()\n",
    "        optimizer.step()\n",
    "\n",
    "        total_loss += loss.item()\n",
    "\n",
    "    print(f\"Epoch {epoch + 1}, Loss: {total_loss / len(train_data_loader)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 321,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Mean Absolute Error on Test Set: 1.6231505796313286\n"
     ]
    }
   ],
   "source": [
    "# Evaluation loop\n",
    "model.eval()\n",
    "total_mae = 0.0\n",
    "\n",
    "with torch.no_grad():\n",
    "    for batch_x, batch_lengths, batch_y in test_data_loader:\n",
    "        batch_x = torch.nn.functional.one_hot(batch_x, num_classes=6).float()\n",
    "        outputs = model(batch_x, batch_lengths)\n",
    "        mae = torch.mean(torch.abs(outputs - batch_y))\n",
    "        total_mae += mae.item()\n",
    "\n",
    "print(f\"Mean Absolute Error on Test Set: {total_mae / len(test_data_loader)}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 322,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Requirement already satisfied: streamlit in c:\\users\\shubh\\anaconda3\\lib\\site-packages (1.41.1)Note: you may need to restart the kernel to use updated packages.\n",
      "\n",
      "Requirement already satisfied: altair<6,>=4.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (5.5.0)\n",
      "Requirement already satisfied: blinker<2,>=1.0.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (1.6.2)\n",
      "Requirement already satisfied: cachetools<6,>=4.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (5.5.1)\n",
      "Requirement already satisfied: click<9,>=7.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (8.1.7)\n",
      "Requirement already satisfied: numpy<3,>=1.23 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (2.2.1)\n",
      "Requirement already satisfied: packaging<25,>=20 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (24.2)\n",
      "Requirement already satisfied: pandas<3,>=1.4.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (2.2.3)\n",
      "Requirement already satisfied: pillow<12,>=7.1.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (11.0.0)\n",
      "Requirement already satisfied: protobuf<6,>=3.20 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (4.25.5)\n",
      "Requirement already satisfied: pyarrow>=7.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (17.0.0)\n",
      "Requirement already satisfied: requests<3,>=2.27 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (2.32.3)\n",
      "Requirement already satisfied: rich<14,>=10.14.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (13.9.3)\n",
      "Requirement already satisfied: tenacity<10,>=8.1.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (9.0.0)\n",
      "Requirement already satisfied: toml<2,>=0.10.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (0.10.2)\n",
      "Requirement already satisfied: typing-extensions<5,>=4.3.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (4.12.2)\n",
      "Requirement already satisfied: watchdog<7,>=2.1.5 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (4.0.1)\n",
      "Requirement already satisfied: gitpython!=3.1.19,<4,>=3.0.7 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (3.1.44)\n",
      "Requirement already satisfied: pydeck<1,>=0.8.0b4 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (0.9.1)\n",
      "Requirement already satisfied: tornado<7,>=6.0.3 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from streamlit) (6.4.2)\n",
      "Requirement already satisfied: jinja2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (3.1.4)\n",
      "Requirement already satisfied: jsonschema>=3.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (4.23.0)\n",
      "Requirement already satisfied: narwhals>=1.14.2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from altair<6,>=4.0->streamlit) (1.23.0)\n",
      "Requirement already satisfied: colorama in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from click<9,>=7.0->streamlit) (0.4.6)\n",
      "Requirement already satisfied: gitdb<5,>=4.0.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from gitpython!=3.1.19,<4,>=3.0.7->streamlit) (4.0.12)\n",
      "Requirement already satisfied: python-dateutil>=2.8.2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas<3,>=1.4.0->streamlit) (2.9.0.post0)\n",
      "Requirement already satisfied: pytz>=2020.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas<3,>=1.4.0->streamlit) (2024.1)\n",
      "Requirement already satisfied: tzdata>=2022.7 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from pandas<3,>=1.4.0->streamlit) (2023.3)\n",
      "Requirement already satisfied: charset-normalizer<4,>=2 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.3.2)\n",
      "Requirement already satisfied: idna<4,>=2.5 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (3.7)\n",
      "Requirement already satisfied: urllib3<3,>=1.21.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (1.26.19)\n",
      "Requirement already satisfied: certifi>=2017.4.17 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from requests<3,>=2.27->streamlit) (2024.12.14)\n",
      "Requirement already satisfied: markdown-it-py>=2.2.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.2.0)\n",
      "Requirement already satisfied: pygments<3.0.0,>=2.13.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from rich<14,>=10.14.0->streamlit) (2.15.1)\n",
      "Requirement already satisfied: smmap<6,>=3.0.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from gitdb<5,>=4.0.1->gitpython!=3.1.19,<4,>=3.0.7->streamlit) (5.0.2)\n",
      "Requirement already satisfied: MarkupSafe>=2.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jinja2->altair<6,>=4.0->streamlit) (2.1.3)\n",
      "Requirement already satisfied: attrs>=22.2.0 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (24.3.0)\n",
      "Requirement already satisfied: jsonschema-specifications>=2023.03.6 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (2023.7.1)\n",
      "Requirement already satisfied: referencing>=0.28.4 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.30.2)\n",
      "Requirement already satisfied: rpds-py>=0.7.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from jsonschema>=3.0->altair<6,>=4.0->streamlit) (0.10.6)\n",
      "Requirement already satisfied: mdurl~=0.1 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from markdown-it-py>=2.2.0->rich<14,>=10.14.0->streamlit) (0.1.0)\n",
      "Requirement already satisfied: six>=1.5 in c:\\users\\shubh\\anaconda3\\lib\\site-packages (from python-dateutil>=2.8.2->pandas<3,>=1.4.0->streamlit) (1.17.0)\n"
     ]
    }
   ],
   "source": [
    "pip install streamlit\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 323,
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 354,
   "metadata": {},
   "outputs": [],
   "source": [
    "def set_seed(seed=42):\n",
    "    torch.manual_seed(seed)\n",
    "    random.seed(seed)\n",
    "    np.random.seed(seed)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 356,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "def sequence_to_tensor(seq):\n",
    "    \"\"\"Convert a DNA sequence to a tensor of integers.\"\"\"\n",
    "    int_seq = [dna2int[char] for char in seq]\n",
    "    return torch.tensor([int_seq], dtype=torch.long)  "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 358,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2025-01-24 22:59:20.643 WARNING streamlit.runtime.scriptrunner_utils.script_run_context: Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.344 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\shubh\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n",
      "2025-01-24 22:59:21.345 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.347 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.350 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.353 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.356 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.358 Session state does not function when running a script without `streamlit run`\n",
      "2025-01-24 22:59:21.363 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n",
      "2025-01-24 22:59:21.365 Thread 'MainThread': missing ScriptRunContext! This warning can be ignored when running in bare mode.\n"
     ]
    }
   ],
   "source": [
    "st.title(\"CpG Detector\")\n",
    "\n",
    "sequence = st.text_input(\"Enter a DNA sequence (e.g., 'ACGCGTGACG'):\")\n",
    "\n",
    "if sequence:\n",
    "    try:\n",
    "        # Preprocess input\n",
    "        input_tensor = sequence_to_tensor(sequence)\n",
    "\n",
    "        # Model prediction\n",
    "        with torch.no_grad():\n",
    "            prediction = model(input_tensor).item()\n",
    "\n",
    "        # Output result\n",
    "        st.write(f\"Predicted number of CpGs: {prediction:.2f}\")\n",
    "        st.write(f\"Actual CpG count: {count_cpgs(sequence)}\")\n",
    "    except KeyError:\n",
    "        st.error(\"Invalid DNA sequence. Please use only characters: N, A, C, G, T.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "collapsed_sections": [],
   "name": "Xi Yangs Copy of broken-nn-template.ipynb",
   "provenance": [
    {
     "file_id": "13GlbI_pdKNES8I718iwl1KNnMZ73iOOn",
     "timestamp": 1651680757732
    }
   ]
  },
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
