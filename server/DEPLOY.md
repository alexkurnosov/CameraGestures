# Deploying the Training Server to a VPS

Step-by-step guide for running the CameraGestures training server on a fresh VPS.

---

## Prerequisites

A fresh Ubuntu/Debian VPS needs Docker and Git. Run these once:

```bash
# Install Docker Engine + Compose plugin
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER   # allow running docker without sudo
newgrp docker                   # apply group change in current shell

# Install Git (usually pre-installed)
sudo apt-get install -y git
```

Verify:

```bash
docker --version        # Docker version 24+
docker compose version  # Docker Compose version 2+
git --version
```

---

## 1. Clone the repository

```bash
git clone https://github.com/alexkurnosov/CameraGestures.git
cd CameraGestures/server
```

---

## 2. Configure the environment

Run the interactive setup script. It walks through every config variable with
defaults, auto-generates `JWT_SECRET`, and lets you set or auto-generate
`REGISTRATION_TOKEN`:

```bash
bash setup_env.sh
```

The script will:
- Prompt for each setting — press **Enter** to keep the default
- Auto-generate `JWT_SECRET` (you never need to type or remember it)
- Ask for a `REGISTRATION_TOKEN` passphrase — press **Enter** to generate one
- Print both credentials to the console when done
- Back up any existing `.env` before overwriting it
- Write `.env` with permissions `600` (owner-readable only)

> **Save the printed `REGISTRATION_TOKEN`.** You will enter it in the iOS app.
> `JWT_SECRET` lives only in `.env` — you do not need to store it elsewhere.

---

## 3. Start the server

```bash
docker compose up --build -d
```

`-d` runs it in the background. Logs are available with:

```bash
docker compose logs -f
```

---

## 4. Verify it is running

```bash
curl http://localhost:8000/health
# → {"status":"ok"}
```

The interactive API docs are at `http://localhost:8000/docs`.

---

## 5. Open the firewall port

If you are **not** using the Caddy HTTPS proxy (plain HTTP, private network):

```bash
sudo ufw allow 8000
sudo ufw status
```

If you are using Caddy (recommended for public VPS — see step 7), open ports
80 and your chosen HTTPS port instead. **Do not open 8000** — the app is bound
to `127.0.0.1` and is only reachable via the Caddy proxy:

```bash
sudo ufw allow 80       # ACME HTTP-01 certificate issuance
sudo ufw allow 9443     # HTTPS (replace 9443 with your chosen port)
sudo ufw status
```

If your VPS provider has a separate network firewall panel (AWS Security Groups,
DigitalOcean Firewall, Hetzner Firewall, etc.), add inbound TCP rules for the
same ports there as well.

---

## 6. Configure the iOS app

1. Open the **ModelTrainingApp** on your device
2. Go to **Settings → Server**
3. Set **Server URL** to `http://<your-vps-ip>:8000`
4. Set **Registration Token** to the value printed by `setup_env.sh`
5. The app registers automatically on the first request — no extra step needed

---

## 7. (Optional) HTTPS with Caddy

For a public VPS, HTTPS is recommended. Caddy handles TLS certificates
automatically. This setup runs Caddy as an additional Docker service alongside
the app, so no separate installation is required.

Port 443 is occupied by another service, so Caddy listens on a configurable
port (default **9443**). Certificates are issued via the ACME HTTP-01 challenge
on port 80, which must be free and reachable from the internet.

### 7a. Generate the Caddyfile

```bash
bash setup_caddy.sh
```

The script will:
- Ask for your **domain name** (required — Caddy's automatic TLS needs a domain)
- Ask for the **HTTPS port** (default `9443`)
- Write a `Caddyfile` configured for your domain
- Add `HTTPS_PORT` to your `.env`
- Print the firewall commands and the iOS Server URL to use

### 7b. Open firewall ports

```bash
sudo ufw allow 80      # ACME HTTP-01 — must be reachable from the internet
sudo ufw allow 9443    # HTTPS (replace with your chosen port)
```

Apply the same rules in your VPS provider's network firewall panel if it has one.

### 7c. Start with Caddy

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
```

Caddy fetches a TLS certificate automatically on first startup. Check its logs
if the certificate does not appear within a minute:

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml logs caddy -f
```

### 7d. Update the iOS app

Set **Server URL** to `https://your-domain.com:9443` (replace port if you chose
a different one).

> iOS requires HTTPS for arbitrary network connections by default. If you use
> plain HTTP you must add an `NSExceptionDomain` entry for the VPS IP/hostname
> in the app's `Info.plist`.

---

## Stopping and restarting

Without Caddy:
```bash
docker compose down          # stop (data is preserved in ./data volume)
docker compose up -d         # start again
docker compose restart       # restart without rebuilding
docker compose up --build -d # rebuild image after a git pull
```

With Caddy (append `-f docker-compose.caddy.yml` to every command):
```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml down
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up -d
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
```

---

## Updating to a new version

```bash
git pull
cd server
# Without Caddy:
docker compose up --build -d
# With Caddy:
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
```

Your examples and trained models are stored in `./data/` which is mounted as a
Docker volume and is never touched by `git pull`.

---

## Revoking a device or rotating secrets

| Goal | Action |
|------|--------|
| Revoke one device | Delete its row from the `devices` table in `data/gestures.db` |
| Invalidate **all** tokens | Change `JWT_SECRET` in `.env` and restart (`docker compose restart`) |
| Change the registration passphrase | Update `REGISTRATION_TOKEN` in `.env` and restart — existing tokens remain valid, new registrations need the new passphrase |
