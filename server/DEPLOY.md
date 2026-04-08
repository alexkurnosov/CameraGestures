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

```bash
# UFW (Ubuntu default)
sudo ufw allow 8000
sudo ufw status
```

If your VPS provider has a separate network firewall panel (AWS Security Groups,
DigitalOcean Firewall, Hetzner Firewall, etc.), add an inbound TCP rule for
port 8000 there as well.

---

## 6. Configure the iOS app

1. Open the **ModelTrainingApp** on your device
2. Go to **Settings → Server**
3. Set **Server URL** to `http://<your-vps-ip>:8000`
4. Set **Registration Token** to the value printed by `setup_env.sh`
5. The app registers automatically on the first request — no extra step needed

---

## 7. (Optional) HTTPS with a reverse proxy

Exposing plain HTTP on port 8000 is acceptable on a private network, but for a
public VPS HTTPS is recommended. The quickest way is **Caddy** — it handles
TLS certificates automatically.

```bash
sudo apt-get install -y caddy
```

`/etc/caddy/Caddyfile`:

```
your-domain.com {
    reverse_proxy localhost:8000
}
```

```bash
sudo systemctl restart caddy
```

With Caddy in front, keep port 8000 **closed** in the firewall and only expose
443. Update the iOS **Server URL** to `https://your-domain.com`.

iOS requires HTTPS for arbitrary network connections by default. If you use
plain HTTP you must add an `NSExceptionDomain` entry for the VPS IP/hostname in
the app's `Info.plist`.

---

## Stopping and restarting

```bash
docker compose down          # stop (data is preserved in ./data volume)
docker compose up -d         # start again
docker compose restart       # restart without rebuilding
docker compose up --build -d # rebuild image after a git pull
```

---

## Updating to a new version

```bash
git pull
cd server
docker compose up --build -d
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
