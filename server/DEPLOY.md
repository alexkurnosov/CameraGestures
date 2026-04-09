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

If you are using Caddy (recommended for public VPS — see step 7), only open
your chosen HTTPS port. **Do not open 8000 or 80** — the app is bound to
`127.0.0.1`, and certificates are issued via Cloudflare DNS-01 (no port 80
needed):

```bash
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

Port 443 is occupied by another service. Caddy therefore listens on a
configurable port (default **9443**) and issues certificates via the
**Cloudflare DNS-01 challenge** — ports 80 and 443 do not need to be free.

### 7a. Create a Cloudflare API token

1. Go to <https://dash.cloudflare.com/profile/api-tokens>
2. Click **Create Token → Use template: Edit zone DNS**
3. Under **Zone Resources** select your domain (`akthesnp.com`)
4. Create the token and copy it — you will paste it into `setup_caddy.sh`

### 7b. Run the setup script

```bash
bash setup_caddy.sh
```

The script will:
- Ask for your **domain name** (e.g. `cameragesturesmodeltrain.akthesnp.com`)
- Ask for the **HTTPS port** (default `9443`)
- Ask for the **Cloudflare API token** (input is hidden)
- Write the `Caddyfile` configured for DNS-01
- Add `HTTPS_PORT` and `CF_API_TOKEN` to your `.env`

### 7c. Open the firewall

Only the HTTPS port needs to be open — no port 80 required:

```bash
sudo ufw allow 9443    # replace with your chosen port
```

Apply the same rule in your VPS provider's network firewall panel if it has one.

### 7d. Start with Caddy

The first run builds a custom Caddy image with the Cloudflare DNS plugin
(~1 minute):

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml up --build -d
```

Caddy fetches a TLS certificate automatically. Check its logs if the
certificate does not appear within a minute:

```bash
docker compose -f docker-compose.yml -f docker-compose.caddy.yml logs caddy -f
```

### 7e. Update the iOS app

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
